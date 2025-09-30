#!/usr/bin/env python3
"""
Setup training pipeline with the extracted Giramille dataset.
"""

import os
import shutil
import json
from pathlib import Path
import argparse

def setup_training_environment():
    """Set up the training environment with the extracted dataset."""
    
    # Create necessary directories
    dirs_to_create = [
        'data/train',
        'data/val', 
        'data/test',
        'data/processed',
        'checkpoints',
        'logs',
        'outputs/generated',
        'outputs/vectors'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Copy dataset images to training directory
    source_dir = Path('sample_dataset/images')
    train_dir = Path('data/train')
    
    if source_dir.exists():
        image_files = list(source_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff,webp}'))
        print(f"Found {len(image_files)} images in dataset")
        
        # Copy all images to train directory
        for img_file in image_files:
            dest_file = train_dir / img_file.name
            shutil.copy2(img_file, dest_file)
        
        print(f"Copied {len(image_files)} images to {train_dir}")
        
        # Create a simple train/val split (80/20)
        val_dir = Path('data/val')
        all_images = list(train_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff,webp}'))
        
        # Move 20% to validation
        val_count = len(all_images) // 5
        for i, img_file in enumerate(all_images[:val_count]):
            dest_file = val_dir / img_file.name
            shutil.move(str(img_file), str(dest_file))
        
        print(f"Moved {val_count} images to validation set")
        print(f"Training set: {len(list(train_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff,webp}')))} images")
        print(f"Validation set: {len(list(val_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff,webp}')))} images")
        
    else:
        print(f"Warning: Dataset directory {source_dir} not found")
    
    # Create dataset manifest
    create_dataset_manifest()
    
    # Create training configuration
    create_training_config()
    
    print("\n✅ Training environment setup complete!")
    print("\nNext steps:")
    print("1. Run: python scripts/prepare_dataset.py --data-dir data/train --out-dir data/processed")
    print("2. Run: python train.py --config configs/training_config.yaml")
    print("3. Run: python infer.py --checkpoint checkpoints/best_model.pth --prompt 'a red bird'")

def create_dataset_manifest():
    """Create a manifest of the dataset."""
    
    manifest = {
        "dataset_name": "Giramille Sample Dataset",
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "image_formats": ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"],
        "categories": {
            "animals": ["bird", "cat", "dog", "fish", "butterfly", "horse", "bear", "chick", "ant", "frog", "crocodile", "moose", "t-rex", "fairy", "witch"],
            "objects": ["car", "airplane", "train", "bus", "boat", "house", "castle", "tree", "flower", "hat", "star", "heart", "sun", "moon", "cloud", "ball", "book", "cup"],
            "scenes": ["forest", "beach", "mountain", "city", "school", "farm", "prison", "stage", "park", "bridge", "statue", "sky", "ground", "wood", "water", "rail"],
            "food": ["apple", "bread", "milk", "banana", "ice cream", "fish food"],
            "characters": ["giramille", "indian", "firefighter", "chef"],
            "items": ["wand", "fishing rod", "surfboard", "mask", "flag", "map", "leaf", "rainbow", "clothespin", "belt", "tutu", "bow", "frame", "sign"],
            "colors": ["red", "blue", "green", "yellow", "purple", "pink", "brown", "black", "white", "orange"],
            "holidays": ["christmas", "easter", "birthday", "congratulations"],
            "hygiene": ["shampoo", "soap", "toothbrush", "mouthwash", "dental floss", "diaper", "diaper cream", "conditioner", "wet wipes", "hand sanitizer"]
        },
        "metadata": {
            "created": "2025-09-20",
            "source": "7. Banco de Imagens.zip",
            "extracted_images": 321,
            "description": "Sample dataset from Giramille project containing various characters, objects, and scenes"
        }
    }
    
    # Count images in each split
    train_dir = Path('data/train')
    val_dir = Path('data/val')
    test_dir = Path('data/test')
    
    if train_dir.exists():
        manifest["train_images"] = len(list(train_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff,webp}')))
    if val_dir.exists():
        manifest["val_images"] = len(list(val_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff,webp}')))
    if test_dir.exists():
        manifest["test_images"] = len(list(test_dir.glob('*.{jpg,jpeg,png,gif,bmp,tiff,webp}')))
    
    manifest["total_images"] = manifest["train_images"] + manifest["val_images"] + manifest["test_images"]
    
    # Save manifest
    with open('data/dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Created dataset manifest: data/dataset_manifest.json")

def create_training_config():
    """Create training configuration file."""
    
    config = {
        "model": {
            "name": "diffusion_unet",
            "input_channels": 3,
            "output_channels": 3,
            "base_channels": 64,
            "num_layers": 4,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "save_every": 10,
            "validate_every": 5,
            "mixed_precision": True,
            "gradient_checkpointing": True
        },
        "data": {
            "train_dir": "data/train",
            "val_dir": "data/val",
            "test_dir": "data/test",
            "processed_dir": "data/processed",
            "image_size": [512, 512],
            "augmentation": {
                "horizontal_flip": 0.5,
                "rotation": 15,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2
            }
        },
        "optimizer": {
            "type": "AdamW",
            "weight_decay": 1e-4,
            "betas": [0.9, 0.999]
        },
        "scheduler": {
            "type": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1e-6
        },
        "loss": {
            "type": "mse",
            "weight": 1.0
        },
        "checkpoint": {
            "save_dir": "checkpoints",
            "best_metric": "val_loss",
            "save_top_k": 3
        },
        "logging": {
            "log_dir": "logs",
            "log_every": 100,
            "save_images_every": 500
        }
    }
    
    # Save config
    with open('configs/training_config.yaml', 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Created training config: configs/training_config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup training environment")
    parser.add_argument("--force", action="store_true", help="Force recreation of directories")
    args = parser.parse_args()
    
    setup_training_environment()
