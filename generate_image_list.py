#!/usr/bin/env python3
"""
Scan dataset/train/ and output data_lists/all_images.csv
Columns: file,label,score (score 为随机代理数)
file 字段仅保存文件名，与 train_model.py 的映射保持一致
"""
import os
import csv
import pathlib
import random
DATASET_DIR = "dataset/train"
OUT_DIR = "data_lists"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = f"{OUT_DIR}/all_images.csv"

rows = []
for pose in sorted(os.listdir(DATASET_DIR)):
    pose_dir = pathlib.Path(DATASET_DIR, pose)
    if not pose_dir.is_dir():
        continue
    for img in pose_dir.rglob("*"):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            rows.append(
                {
                    "file": img.name,
                    "label": pose,
                    "score": round(random.uniform(0.7, 1.0), 3),
                }
            )

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "label", "score"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ CSV 生成完毕，共 {len(rows)} 条 -> {CSV_PATH}")

