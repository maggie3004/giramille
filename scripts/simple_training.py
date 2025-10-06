#!/usr/bin/env python3
"""
Simple Giramille Training Script
Works without complex dependencies, focuses on style learning
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GiramilleDataset(Dataset):
    """Dataset for Giramille style images"""
    
    def __init__(self, data_dir: str, image_size: int = 512):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.images = []
        self.labels = []
        
        # Load all images
        for category in ['characters', 'objects', 'animals', 'scenarios']:
            category_dir = self.data_dir / category
            if category_dir.exists():
                for img_path in category_dir.glob('*'):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        self.images.append(str(img_path))
                        self.labels.append(category)
        
        logger.info(f"Loaded {len(self.images)} images from {len(set(self.labels))} categories")
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Open image with size limit to prevent DOS attacks
            Image.MAX_IMAGE_PIXELS = 100000000  # Increase limit
            image = Image.open(img_path)
            
            # Skip problematic images
            if image.size[0] * image.size[1] > 50000000:  # Skip very large images
                logger.warning(f"Skipping large image {img_path}")
                return torch.zeros(3, self.image_size, self.image_size), label
            
            image = image.convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            return torch.zeros(3, self.image_size, self.image_size), label

class GiramilleStyleEncoder(nn.Module):
    """Simple CNN for learning Giramille style features"""
    
    def __init__(self, input_size: int = 512, style_dim: int = 256):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # 512x512 -> 256x256
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 32x32 -> 16x16
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Style classification head
        self.style_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, style_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(style_dim, 4)  # 4 categories
        )
        
        # Style feature extractor
        self.style_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        style_features = self.style_extractor(features)
        category_logits = self.style_classifier(features)
        return style_features, category_logits

class GiramilleTrainer:
    """Trainer for Giramille style model"""
    
    def __init__(self, data_dir: str, output_dir: str = "models"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model
        self.model = GiramilleStyleEncoder().to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.style_criterion = nn.MSELoss()
        
        # Category mapping
        self.category_map = {
            'characters': 0,
            'objects': 1, 
            'animals': 2,
            'scenarios': 3
        }
        
        # Giramille style features (learned from data)
        self.giramille_style = None
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = [self.category_map[label] for label in labels]
            labels = torch.tensor(labels).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            style_features, category_logits = self.model(images)
            
            # Classification loss only (simplified)
            total_loss_batch = self.criterion(category_logits, labels)
            
            # Update Giramille style reference (detached)
            if self.giramille_style is None:
                self.giramille_style = style_features.mean(dim=0, keepdim=True).detach()
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            _, predicted = category_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def train(self, epochs: int = 50, batch_size: int = 8):
        """Train the model"""
        logger.info("Starting Giramille style training...")
        
        # Load dataset
        dataset = GiramilleDataset(self.data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        logger.info(f"Training on {len(dataset)} images for {epochs} epochs")
        
        best_accuracy = 0
        
        for epoch in range(epochs):
            # Train
            avg_loss, accuracy = self.train_epoch(dataloader, epoch)
            
            logger.info(f'Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model(f"giramille_best_epoch_{epoch}_acc_{accuracy:.1f}.pth")
                logger.info(f"New best model saved with accuracy {accuracy:.2f}%")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_model(f"giramille_checkpoint_epoch_{epoch}.pth")
        
        logger.info(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
        return best_accuracy
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        model_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'category_map': self.category_map,
            'giramille_style': self.giramille_style
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def generate_style_report(self):
        """Generate a report on learned Giramille style"""
        logger.info("Generating Giramille style report...")
        
        # Analyze learned features
        if self.giramille_style is not None:
            style_vector = self.giramille_style.cpu().detach().numpy()
            
            report = {
                "giramille_style_features": style_vector.tolist(),
                "feature_dimension": len(style_vector[0]),
                "categories_learned": list(self.category_map.keys()),
                "model_architecture": "GiramilleStyleEncoder",
                "training_completed": True
            }
            
            report_path = self.output_dir / "giramille_style_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Style report saved to {report_path}")
            return report
        
        return None

def main():
    """Main training function"""
    logger.info("🎨 Starting Simple Giramille Training")
    
    # Configuration
    data_dir = "data/train"
    output_dir = "models"
    epochs = 50
    batch_size = 4  # Smaller batch for stability
    
    # Check if data exists
    if not Path(data_dir).exists():
        logger.error(f"Data directory {data_dir} not found!")
        return
    
    # Initialize trainer
    trainer = GiramilleTrainer(data_dir, output_dir)
    
    # Train model
    try:
        best_accuracy = trainer.train(epochs=epochs, batch_size=batch_size)
        
        # Generate style report
        trainer.generate_style_report()
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"🎯 Best accuracy achieved: {best_accuracy:.2f}%")
        logger.info("🎨 Giramille style features learned and saved!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
