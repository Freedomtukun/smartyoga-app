#!/usr/bin/env python3
"""
Yoga Pose Score Regression Model Training Script

This script:
1. Loads training data from CSV and image directories (supports subdirectories)
2. Trains a CNN regression model to predict pose scores
3. Automatically cleans up training data after completion
4. Supports logging, multi-threading, and visualization
5. Sends email notifications upon completion

Usage:
    python train_model.py [--epochs N] [--batch-size N] [--workers N] [--multi-head] [--email-pass PASS]
"""

import os
import sys
import subprocess
import logging
import argparse
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
# Pillow ≥ 9.1 用 Image.Resampling；低版本回退到 Image.LANCZOS
from PIL import Image
try:
    _RESAMPLE = Image.Resampling.LANCZOS        # Pillow ≥ 9.1
except AttributeError:                          # Pillow < 9.1
    _RESAMPLE = Image.LANCZOS
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# 🔄 unified dataset migration
DATASET_DIR = "dataset/train"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Configure logging
def setup_logging():
    """Setup logging configuration with both file and console output."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__), log_file


# Global logger and log file
logger, LOG_FILE = setup_logging()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train yoga pose score regression model")
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for image loading (default: 4)')
    parser.add_argument('--multi-head', action='store_true', help='Use multi-head model for regression + classification')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Validation split ratio (default: 0.1)')
    parser.add_argument('--email-pass', type=str, default=None, help='Gmail password for notifications (or set GOOGLE_MAIL_PASS env var)')
    
    return parser.parse_args()


def load_and_preprocess_image(
    image_path: Path,
    target_size: Tuple[int, int] = (224, 224),
) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array normalized to [0, 1], or None if loading fails
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, _RESAMPLE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def load_image_with_metadata(img_path: Path, file_to_data: dict, is_correct: bool) -> Optional[Tuple]:
    """
    Load a single image with its metadata.
    
    Args:
        img_path: Path to image file
        file_to_data: Dictionary mapping filename to (score, label)
        is_correct: Whether this is from the correct directory
    
    Returns:
        Tuple of (image_array, score, binary_label) or None
    """
    filename = img_path.name
    if filename in file_to_data:
        img_array = load_and_preprocess_image(img_path)
        if img_array is not None:
            score = file_to_data[filename][0]
            binary_label = 1 if is_correct else 0
            return (img_array, score, binary_label)
    return None


def load_training_data(workers: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data from ``DATASET_DIR`` recursively.

    This version ignores the old ``training_data`` structure and assumes all
    images reside under ``DATASET_DIR/<pose_id>``. Each image is treated as a
    correct sample with score ``1.0``.
    """
    logger.info("=" * 60)
    logger.info("Starting data loading process...")

    # 确保 CSV 存在；若缺失则自动生成
    csv_path = "data_lists/all_images.csv"
    if not os.path.exists(csv_path):
        logger.info("CSV 不存在，调用 generate_image_list.py 重新生成…")
        subprocess.run([sys.executable, "generate_image_list.py"], check=True)

    # 映射：文件名 → 分数；若无分数列则默认 1.0
    score_map = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "score" in df.columns:
            score_map = {row["file"]: float(row["score"]) for _, row in df.iterrows()}
        logger.info(
            f"Loaded CSV ({len(df)} rows) — score column: {'yes' if score_map else 'no'}"
        )

    dataset_path = Path(DATASET_DIR)
    image_paths: List[Path] = []
    for pose_dir in sorted(dataset_path.iterdir()):
        if not pose_dir.is_dir():
            continue
        for img_path in pose_dir.rglob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_paths.append(img_path)

    logger.info(f"Found {len(image_paths)} images in {DATASET_DIR}")
    logger.info(f"Loading images using {workers} workers...")

    images, scores, binary_labels = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(load_and_preprocess_image, p): p for p in image_paths
        }
        loaded_count = 0
        for future in as_completed(future_to_path):
            img_array = future.result()
            img_path = future_to_path[future]
            if img_array is not None:
                images.append(img_array)
                # 真实分数或默认 1.0
                scores.append(score_map.get(img_path.name, 1.0))
                binary_labels.append(1.0)  # 目前只有正样本
                loaded_count += 1
                if loaded_count % 100 == 0:
                    logger.info(
                        f"Loaded {loaded_count}/{len(image_paths)} images..."
                    )

    if not images:
        raise ValueError("No images were successfully loaded!")

    images = np.array(images, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    binary_labels = np.array(binary_labels, dtype=np.float32)

    logger.info(f"Successfully loaded {len(images)} images")
    logger.info(f"Images shape: {images.shape}")

    return images, scores, binary_labels


def create_cnn_model(input_shape: Tuple[int, int, int], learning_rate: float = 0.001, 
                     use_multi_head: bool = False) -> keras.Model:
    """
    Create a CNN model for regression (and optionally classification).
    
    Args:
        input_shape: Input image shape (height, width, channels)
        learning_rate: Learning rate for optimizer
        use_multi_head: Whether to create a multi-head model with both regression and classification outputs
    
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Convolutional base
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Flatten and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    if use_multi_head:
        # Multi-head output
        score_output = layers.Dense(1, activation='linear', name='score')(x)
        class_output = layers.Dense(1, activation='sigmoid', name='classification')(x)
        
        model = keras.Model(inputs=inputs, outputs=[score_output, class_output])
        
        # Compile with multiple losses
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'score': 'mse',
                'classification': 'binary_crossentropy'
            },
            loss_weights={
                'score': 1.0,
                'classification': 0.5
            },
            metrics={
                'score': ['mae'],
                'classification': ['accuracy']
            }
        )
    else:
        # Single regression output
        score_output = layers.Dense(1, activation='linear', name='score')(x)
        
        model = keras.Model(inputs=inputs, outputs=score_output)
        
        # Compile for regression
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    return model


def plot_training_history(history, use_multi_head: bool = False, plots_dir: str = "training_plots"):
    """
    Plot and save training history graphs.

    Args:
        history: Keras training history object or history dictionary
        use_multi_head: Whether this was a multi-head model
        plots_dir: Directory to save the plots
    """
    logger.info("Generating training visualization...")
    
    # Create output directory
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Support both History objects and plain dicts
    history_data = history.history if hasattr(history, "history") else history
    
    if use_multi_head:
        # Create a figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Score Loss
        ax1.plot(history_data['score_loss'], label='Train Score Loss')
        ax1.plot(history_data['val_score_loss'], label='Val Score Loss')
        ax1.set_title('Score Regression Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Score MAE
        ax2.plot(history_data['score_mae'], label='Train Score MAE')
        ax2.plot(history_data['val_score_mae'], label='Val Score MAE')
        ax2.set_title('Score Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        # Classification Loss
        ax3.plot(history_data['classification_loss'], label='Train Class Loss')
        ax3.plot(history_data['val_classification_loss'], label='Val Class Loss')
        ax3.set_title('Classification Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Binary Crossentropy')
        ax3.legend()
        ax3.grid(True)
        
        # Classification Accuracy
        ax4.plot(history_data['classification_accuracy'], label='Train Accuracy')
        ax4.plot(history_data['val_classification_accuracy'], label='Val Accuracy')
        ax4.set_title('Classification Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True)
        
    else:
        # Create a figure with 1x2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss
        ax1.plot(history_data['loss'], label='Train Loss')
        ax1.plot(history_data['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss (MSE)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(history_data['mae'], label='Train MAE')
        ax2.plot(history_data['val_mae'], label='Val MAE')
        ax2.set_title('Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"training_history_{timestamp}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    logger.info(f"Training plots saved to: {plot_path}")


def train_model(images: np.ndarray, scores: np.ndarray, binary_labels: np.ndarray,
                epochs: int = 2, batch_size: int = 16, learning_rate: float = 0.001,
                validation_split: float = 0.1, use_multi_head: bool = False) -> Tuple[keras.Model, dict]:
    """
    Train the CNN model.
    
    Args:
        images: Array of preprocessed images
        scores: Array of score labels
        binary_labels: Array of binary labels (correct/incorrect)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        validation_split: Validation split ratio
        use_multi_head: Whether to use multi-head model
    
    Returns:
        Tuple of (trained model, training history)
    """
    logger.info("=" * 60)
    logger.info("Starting model training...")
    
    # 若只有 1 个类别则不能 stratify
    stratifier = binary_labels if len(np.unique(binary_labels)) >= 2 else None
    X_train, X_val, y_score_train, y_score_val, y_class_train, y_class_val = train_test_split(
        images,
        scores,
        binary_labels,
        test_size=validation_split,
        random_state=42,
        stratify=stratifier,
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Create model
    input_shape = (224, 224, 3)
    effective_multi_head = use_multi_head and len(np.unique(binary_labels)) >= 2
    model = create_cnn_model(
        input_shape,
        learning_rate=learning_rate,
        use_multi_head=effective_multi_head,
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    # Prepare data for training
    if effective_multi_head:
        y_train = {'score': y_score_train, 'classification': y_class_train}
        y_val = {'score': y_score_val, 'classification': y_class_val}
    else:
        y_train = y_score_train
        y_val = y_score_val
    
    logger.info(f"Training parameters:")
    logger.info(f"- Batch size: {batch_size}")
    logger.info(f"- Epochs: {epochs}")
    logger.info(f"- Learning rate: {learning_rate}")
    logger.info(f"- Multi-head: {effective_multi_head}")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Print final metrics
    logger.info("Training completed!")
    if effective_multi_head:
        logger.info(f"Final training score MAE: {history.history['score_mae'][-1]:.4f}")
        logger.info(f"Final validation score MAE: {history.history['val_score_mae'][-1]:.4f}")
        logger.info(f"Final training classification accuracy: {history.history['classification_accuracy'][-1]:.4f}")
        logger.info(f"Final validation classification accuracy: {history.history['val_classification_accuracy'][-1]:.4f}")
    else:
        logger.info(f"Final training MAE: {history.history['mae'][-1]:.4f}")
        logger.info(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")
    
    return model, history


def save_model(model: keras.Model, model_path: str = "models/yoga_pose_score_regression.h5"):
    """
    Save the trained model.
    
    Args:
        model: Trained Keras model
        model_path: Path to save the model
    """
    logger.info("=" * 60)
    logger.info("Saving model...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Also save in SavedModel format for better compatibility
    saved_model_path = model_path.replace('.h5', '_saved_model')
    model.save(saved_model_path, save_format='tf')
    logger.info(f"Model also saved in SavedModel format to: {saved_model_path}")


def clear_training_data():
    """
    Clear training data by calling the cleanup script.
    """
    logger.info("=" * 60)
    logger.info("Clearing training data...")
    
    cleanup_script = "cos_tools/clear_dataset_images.py"
    
    if not os.path.exists(cleanup_script):
        logger.warning(f"Cleanup script not found at {cleanup_script}")
        logger.warning("Training data will not be cleared automatically.")
        return
    
    try:
        # Call the cleanup script
        result = subprocess.run([sys.executable, cleanup_script], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        logger.info("Training data cleared successfully!")
        if result.stdout:
            logger.info(f"Cleanup output: {result.stdout}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clear training data!")
        logger.error(f"Error message: {e.stderr}")
        logger.error("Please manually run the cleanup script or clear the data.")
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")


def send_email_notification(success: bool, error_message: str = "", gmail_pass: str = None):
    """
    Send email notification about training completion.
    
    Args:
        success: Whether training completed successfully
        error_message: Error message if training failed
        gmail_pass: Gmail password (if not provided, uses GOOGLE_MAIL_PASS env var)
    """
    try:
        # Get Gmail credentials
        if gmail_pass is None:
            gmail_pass = os.environ.get('GOOGLE_MAIL_PASS')
        
        if not gmail_pass:
            logger.warning("No Gmail password provided. Skipping email notification.")
            return
        
        # Email configuration
        sender_email = "canadaziyou@gmail.com"
        receiver_email = "canadaziyou@gmail.com"
        
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        
        # Set subject and body based on success
        if success:
            message["Subject"] = "✅ Yoga Pose Model Training Completed Successfully"
            body = f"""
Training completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!

Training Configuration:
- Log file: {LOG_FILE}
- Model saved to: models/yoga_pose_score_regression.h5
- Training plots saved to: training_plots/

Please check the attached log file for detailed training metrics.
"""
        else:
            message["Subject"] = "❌ Yoga Pose Model Training Failed"
            body = f"""
Training failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

Error Details:
{error_message}

Please check the attached log file for more information.
"""
        
        # Add body to email
        message.attach(MIMEText(body, "plain"))
        
        # Attach log file if it exists
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {os.path.basename(LOG_FILE)}",
            )
            message.attach(part)
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, gmail_pass)
            text = message.as_string()
            server.sendmail(sender_email, receiver_email, text)
        
        logger.info(f"Email notification sent successfully to {receiver_email}")
        
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        # Don't raise - email failure shouldn't stop the script


def main():
    """
    Main training pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("🧘 Yoga Pose Score Regression Model Training")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"- Epochs: {args.epochs}")
    logger.info(f"- Batch size: {args.batch_size}")
    logger.info(f"- Workers: {args.workers}")
    logger.info(f"- Learning rate: {args.learning_rate}")
    logger.info(f"- Validation split: {args.validation_split}")
    logger.info(f"- Multi-head model: {args.multi_head}")
    
    training_success = False
    error_message = ""
    
    try:
        # Load training data
        images, scores, binary_labels = load_training_data(workers=args.workers)
        
        # Train model
        model, history = train_model(
            images, scores, binary_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            use_multi_head=args.multi_head
        )
        
        # Save model
        save_model(model)
        
        # Plot training history
        plot_training_history(history.history, use_multi_head=args.multi_head)
        
        # Clear training data
        clear_training_data()
        
        logger.info("=" * 60)
        logger.info("✅ Training pipeline completed successfully!")
        training_success = True
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        training_success = False
    
    finally:
        # Send email notification
        send_email_notification(
            success=training_success,
            error_message=error_message,
            gmail_pass=args.email_pass
        )
        
        if not training_success:
            sys.exit(1)

def train_from_dataset(
    dataset_dir=DATASET_DIR,
    model_out=None,
    epochs=2,
    batch_size=16,
    workers=4,
    learning_rate=0.001,
    validation_split=0.1,
    use_multi_head=False,
    email_pass=None,
    mode='image_classification',
    plots_dir=None
):
    """
    自动训练管道专用，供云端/定时任务/触发器自动调用，无需人工干预。
    兼容自动训练流程的外部调用接口（不依赖命令行），直接拉通主控脚本

    Args:
        dataset_dir: Dataset directory (currently unused)
        model_out: Optional path to save the trained model
        epochs: Training epochs
        batch_size: Batch size
        workers: Parallel workers for data loading
        learning_rate: Learning rate
        validation_split: Validation split ratio
        use_multi_head: Whether to use multi-head model
        email_pass: Gmail password for notifications
        mode: Training mode ("image_classification" or "sequence_lstm")
        plots_dir: Optional directory to save training plots
    """
    try:
        logger.info("=" * 60)
        logger.info("Auto pipeline trigger: train_from_dataset() called.")

        if mode == 'image_classification':
            images, scores, binary_labels = load_training_data(workers=workers)
            model, history = train_model(
                images, scores, binary_labels,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
                use_multi_head=use_multi_head
            )
        elif mode == 'sequence_lstm':
            raise NotImplementedError("sequence_lstm \u6a21\u5f0f\u6682\u672a\u5b9e\u73b0")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not model_out:
            os.makedirs(f"models/{mode}", exist_ok=True)
            model_out = f"models/{mode}/model_{ts}.h5"
        if not plots_dir:
            os.makedirs(f"training_plots/{mode}", exist_ok=True)
            plots_dir = f"training_plots/{mode}"

        save_model(model, model_path=model_out)
        plot_training_history(history.history, use_multi_head=use_multi_head, plots_dir=plots_dir)
        clear_training_data()
        send_email_notification(
            success=True,
            gmail_pass=email_pass
        )
        logger.info("✅ train_from_dataset finished successfully!")
    except Exception as e:
        logger.error(f"❌ Error in train_from_dataset: {e}")
        send_email_notification(
            success=False,
            error_message=str(e),
            gmail_pass=email_pass
        )
        raise

if __name__ == "__main__":
    main()
