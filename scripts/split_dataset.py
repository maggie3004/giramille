#!/usr/bin/env python3
"""
Split dataset into train/validation sets.
"""

import os
import shutil
import random
from pathlib import Path

def split_dataset(train_dir='data/train', val_dir='data/val', val_ratio=0.2):
    """Split dataset into train and validation sets."""
    
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    # Ensure validation directory exists
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    
    # Get all files and filter by extension
    for file_path in train_path.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                image_files.append(file_path)
    
    print(f'Total images found: {len(image_files)}')
    
    if len(image_files) == 0:
        print("No images found in training directory!")
        return
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    
    val_count = int(len(image_files) * val_ratio)
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    
    print(f'Moving {len(val_files)} images to validation set...')
    
    # Move validation files
    for img_file in val_files:
        dest_file = val_path / img_file.name
        shutil.move(str(img_file), str(dest_file))
    
    # Count final images
    final_train_count = len([f for f in train_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']])
    final_val_count = len([f for f in val_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']])
    
    print(f'Final training set: {final_train_count} images')
    print(f'Final validation set: {final_val_count} images')
    print(f'Total: {final_train_count + final_val_count} images')

if __name__ == "__main__":
    split_dataset()
