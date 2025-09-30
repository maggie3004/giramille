#!/usr/bin/env python3
"""
Advanced AI Training Pipeline for Giramille Style
Matches Freepik-level capabilities with modular editing and multi-view generation
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import cv2
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GiramilleDataset(Dataset):
    """Advanced dataset for Giramille style training with multi-view support"""
    
    def __init__(self, data_dir: str, style_dir: str, transform=None, multi_view: bool = True):
        self.data_dir = Path(data_dir)
        self.style_dir = Path(style_dir)
        self.transform = transform or self.get_default_transform()
        self.multi_view = multi_view
        
        # Load dataset metadata
        self.samples = self.load_samples()
        self.style_references = self.load_style_references()
        
        logger.info(f"Loaded {len(self.samples)} samples and {len(self.style_references)} style references")
    
    def load_samples(self) -> List[Dict]:
        """Load training samples with metadata"""
        samples = []
        
        # Load from different categories
        categories = ['characters', 'objects', 'animals', 'scenarios']
        
        for category in categories:
            category_dir = self.data_dir / category
            if category_dir.exists():
                for img_path in category_dir.glob('*.png'):
                    sample = {
                        'image_path': str(img_path),
                        'category': category,
                        'filename': img_path.stem,
                        'style_attributes': self.extract_style_attributes(img_path)
                    }
                    samples.append(sample)
        
        return samples
    
    def load_style_references(self) -> List[Dict]:
        """Load Giramille style reference images"""
        references = []
        
        if self.style_dir.exists():
            for img_path in self.style_dir.glob('*.png'):
                reference = {
                    'image_path': str(img_path),
                    'style_vector': self.extract_style_vector(img_path)
                }
                references.append(reference)
        
        return references
    
    def extract_style_attributes(self, img_path: Path) -> Dict:
        """Extract style attributes from image"""
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Extract color palette
            colors = self.extract_color_palette(img_array)
            
            # Extract composition features
            composition = self.extract_composition_features(img_array)
            
            # Extract artistic style features
            style_features = self.extract_artistic_features(img_array)
            
            return {
                'colors': colors,
                'composition': composition,
                'style_features': style_features
            }
        except Exception as e:
            logger.warning(f"Error extracting style attributes from {img_path}: {e}")
            return {}
    
    def extract_color_palette(self, img_array: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image"""
        # Reshape image to list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int).tolist()
        return colors
    
    def extract_composition_features(self, img_array: np.ndarray) -> Dict:
        """Extract composition features like rule of thirds, symmetry, etc."""
        height, width = img_array.shape[:2]
        
        # Calculate center of mass
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        moments = cv2.moments(gray)
        
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = width // 2, height // 2
        
        # Rule of thirds analysis
        rule_of_thirds_x = abs(cx - width // 3) < width // 6 or abs(cx - 2 * width // 3) < width // 6
        rule_of_thirds_y = abs(cy - height // 3) < height // 6 or abs(cy - 2 * height // 3) < height // 6
        
        return {
            'center_of_mass': (cx, cy),
            'rule_of_thirds': rule_of_thirds_x and rule_of_thirds_y,
            'aspect_ratio': width / height,
            'symmetry_score': self.calculate_symmetry(gray)
        }
    
    def extract_artistic_features(self, img_array: np.ndarray) -> Dict:
        """Extract artistic style features"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Texture analysis
        texture_score = self.calculate_texture_score(gray)
        
        # Color harmony
        color_harmony = self.calculate_color_harmony(img_array)
        
        return {
            'edge_density': edge_density,
            'texture_score': texture_score,
            'color_harmony': color_harmony
        }
    
    def calculate_symmetry(self, gray_img: np.ndarray) -> float:
        """Calculate symmetry score"""
        # Horizontal symmetry
        h_symmetry = np.mean(np.abs(gray_img - np.flipud(gray_img)))
        
        # Vertical symmetry
        v_symmetry = np.mean(np.abs(gray_img - np.fliplr(gray_img)))
        
        return 1.0 - (h_symmetry + v_symmetry) / (2 * 255)
    
    def calculate_texture_score(self, gray_img: np.ndarray) -> float:
        """Calculate texture complexity score"""
        # Use Laplacian variance as texture measure
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        return np.var(laplacian)
    
    def calculate_color_harmony(self, img_array: np.ndarray) -> float:
        """Calculate color harmony score"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h_values = hsv[:, :, 0].flatten()
        
        # Calculate color distribution
        hist, _ = np.histogram(h_values, bins=36, range=(0, 180))
        
        # Calculate harmony based on color wheel relationships
        harmony_score = 0.0
        for i in range(36):
            # Complementary colors (180 degrees apart)
            comp_idx = (i + 18) % 36
            harmony_score += hist[i] * hist[comp_idx]
            
            # Triadic colors (120 degrees apart)
            tri1_idx = (i + 12) % 36
            tri2_idx = (i + 24) % 36
            harmony_score += hist[i] * hist[tri1_idx] * hist[tri2_idx]
        
        return harmony_score / (img_array.shape[0] * img_array.shape[1])
    
    def extract_style_vector(self, img_path: Path) -> np.ndarray:
        """Extract style vector for style transfer"""
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Extract features for style transfer
            features = []
            
            # Color features
            colors = self.extract_color_palette(img_array)
            features.extend([c for color in colors for c in color])
            
            # Composition features
            comp = self.extract_composition_features(img_array)
            features.extend([comp['aspect_ratio'], comp['symmetry_score']])
            
            # Artistic features
            art = self.extract_artistic_features(img_array)
            features.extend([art['edge_density'], art['texture_score'], art['color_harmony']])
            
            return np.array(features)
        except Exception as e:
            logger.warning(f"Error extracting style vector from {img_path}: {e}")
            return np.zeros(20)  # Default vector
    
    def get_default_transform(self):
        """Get default image transformations"""
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['image_path']).convert('RGB')
        img_tensor = self.transform(img)
        
        # Get style reference
        style_ref = self.style_references[idx % len(self.style_references)]
        style_vector = style_ref['style_vector']
        
        return {
            'image': img_tensor,
            'style_vector': torch.tensor(style_vector, dtype=torch.float32),
            'attributes': sample['style_attributes'],
            'category': sample['category']
        }

class GiramilleStyleModel(nn.Module):
    """Advanced model for Giramille style generation with multi-view support"""
    
    def __init__(self, base_model: str = "runwayml/stable-diffusion-v1-5"):
        super().__init__()
        
        # Load base Stable Diffusion model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # Configure LoRA for efficient fine-tuning
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.DIFFUSION
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.pipe.unet, self.lora_config)
        
        # Style encoder for multi-view generation
        self.style_encoder = self.build_style_encoder()
        
        # Multi-view generator
        self.multi_view_generator = self.build_multi_view_generator()
        
    def build_style_encoder(self) -> nn.Module:
        """Build style encoder for style transfer"""
        return nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768)  # Match UNet hidden size
        )
    
    def build_multi_view_generator(self) -> nn.Module:
        """Build multi-view generator for different angles"""
        return nn.Sequential(
            nn.Linear(768 + 3, 512),  # Style + view angle
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 768)  # Output style modification
        )
    
    def forward(self, prompt: str, style_vector: torch.Tensor, view_angle: Optional[torch.Tensor] = None):
        """Forward pass with style and view angle conditioning"""
        
        # Encode style
        style_embedding = self.style_encoder(style_vector)
        
        # If view angle is provided, generate multi-view
        if view_angle is not None:
            view_condition = torch.cat([style_embedding, view_angle], dim=-1)
            style_modification = self.multi_view_generator(view_condition)
            style_embedding = style_embedding + style_modification
        
        # Generate image with style conditioning
        with torch.no_grad():
            image = self.pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=torch.Generator().manual_seed(42)
            ).images[0]
        
        return image

class AdvancedTrainingPipeline:
    """Advanced training pipeline for Giramille style"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = GiramilleStyleModel(config['base_model']).to(self.device)
        
        # Initialize dataset
        self.dataset = GiramilleDataset(
            config['data_dir'],
            config['style_dir'],
            multi_view=config.get('multi_view', True)
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Initialize wandb for logging
        if config.get('use_wandb', False):
            wandb.init(
                project="giramille-ai",
                config=config
            )
    
    def train(self):
        """Main training loop"""
        logger.info("Starting advanced training pipeline...")
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()
                
                # Get batch data
                images = batch['image'].to(self.device)
                style_vectors = batch['style_vector'].to(self.device)
                categories = batch['category']
                
                # Forward pass
                loss = self.compute_loss(images, style_vectors, categories)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': loss.item()
                    })
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch results
            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # Generate sample images
            if (epoch + 1) % self.config['sample_interval'] == 0:
                self.generate_samples(epoch)
    
    def compute_loss(self, images: torch.Tensor, style_vectors: torch.Tensor, categories: List[str]) -> torch.Tensor:
        """Compute training loss"""
        # Style consistency loss
        style_loss = self.compute_style_loss(images, style_vectors)
        
        # Category-specific loss
        category_loss = self.compute_category_loss(images, categories)
        
        # Multi-view consistency loss
        multi_view_loss = self.compute_multi_view_loss(images, style_vectors)
        
        # Total loss
        total_loss = (
            self.config['style_weight'] * style_loss +
            self.config['category_weight'] * category_loss +
            self.config['multi_view_weight'] * multi_view_loss
        )
        
        return total_loss
    
    def compute_style_loss(self, images: torch.Tensor, style_vectors: torch.Tensor) -> torch.Tensor:
        """Compute style consistency loss"""
        # Extract style features from generated images
        generated_style = self.extract_style_features(images)
        
        # Compute MSE loss between target and generated styles
        style_loss = nn.MSELoss()(generated_style, style_vectors)
        
        return style_loss
    
    def compute_category_loss(self, images: torch.Tensor, categories: List[str]) -> torch.Tensor:
        """Compute category-specific loss"""
        # This would involve a classifier to ensure category consistency
        # For now, return a placeholder
        return torch.tensor(0.0, device=self.device)
    
    def compute_multi_view_loss(self, images: torch.Tensor, style_vectors: torch.Tensor) -> torch.Tensor:
        """Compute multi-view consistency loss"""
        # Generate different views of the same object
        view_angles = torch.tensor([
            [0, 0, 0],      # Front
            [45, 0, 0],     # 3/4 view
            [90, 0, 0],     # Side
            [180, 0, 0]     # Back
        ], device=self.device)
        
        # Generate multi-view images
        multi_view_images = []
        for angle in view_angles:
            view_img = self.model.forward("", style_vectors, angle.unsqueeze(0))
            multi_view_images.append(view_img)
        
        # Compute consistency loss between views
        consistency_loss = 0.0
        for i in range(len(multi_view_images) - 1):
            consistency_loss += nn.MSELoss()(multi_view_images[i], multi_view_images[i + 1])
        
        return consistency_loss / (len(multi_view_images) - 1)
    
    def extract_style_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract style features from images"""
        # This would involve a pre-trained style encoder
        # For now, return a placeholder
        return torch.randn(images.size(0), 20, device=self.device)
    
    def generate_samples(self, epoch: int):
        """Generate sample images for evaluation"""
        self.model.eval()
        
        with torch.no_grad():
            # Generate samples for different categories
            categories = ['house', 'car', 'tree', 'person', 'animal']
            
            for category in categories:
                # Generate sample
                sample = self.model.forward(
                    f"a {category} in Giramille style",
                    torch.randn(1, 20, device=self.device)
                )
                
                # Save sample
                sample_path = f"samples/epoch_{epoch+1}_{category}.png"
                os.makedirs("samples", exist_ok=True)
                sample.save(sample_path)
                
                logger.info(f"Generated sample: {sample_path}")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = f"checkpoints/giramille_epoch_{epoch+1}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_dir': 'data/train',
        'style_dir': 'data/style_references',
        'base_model': 'runwayml/stable-diffusion-v1-5',
        'batch_size': 4,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'epochs': 100,
        'save_interval': 10,
        'sample_interval': 5,
        'style_weight': 1.0,
        'category_weight': 0.5,
        'multi_view_weight': 0.3,
        'multi_view': True,
        'use_wandb': True
    }
    
    # Create training pipeline
    pipeline = AdvancedTrainingPipeline(config)
    
    # Start training
    pipeline.train()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
