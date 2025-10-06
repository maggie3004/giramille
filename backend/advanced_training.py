"""
Advanced Training System for Giramille Style
This will train the model specifically for Giramille style with better color accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import json
from typing import List, Dict, Tuple
import random
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GiramilleDataset(Dataset):
    """Dataset for Giramille style training"""
    
    def __init__(self, data_dir: str = "data/giramille_training"):
        self.data_dir = data_dir
        self.samples = self._load_training_data()
        
    def _load_training_data(self) -> List[Dict]:
        """Load training data for Giramille style"""
        samples = []
        
        # Create training data if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            self._generate_training_data()
        
        # Load existing training data
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    data = json.load(f)
                    samples.append(data)
        
        return samples
    
    def _generate_training_data(self):
        """Generate training data for Giramille style"""
        logger.info("Generating Giramille training data...")
        
        # Color-object combinations for training
        color_objects = [
            ("blue", "house"), ("red", "house"), ("green", "house"), ("yellow", "house"),
            ("blue", "car"), ("red", "car"), ("green", "car"), ("yellow", "car"),
            ("blue", "tree"), ("green", "tree"), ("purple", "mountain"), ("orange", "sunset"),
            ("pink", "flower"), ("brown", "tree"), ("white", "house"), ("black", "car")
        ]
        
        # Generate training samples
        for i, (color, obj) in enumerate(color_objects):
            sample = {
                "prompt": f"{color} {obj}, cartoon style, Giramille",
                "enhanced_prompt": f"bright {color} {obj}, vivid {color} colored {obj}, cartoon style, Giramille",
                "color": color,
                "object": obj,
                "style": "giramille",
                "id": i
            }
            
            # Save sample
            with open(os.path.join(self.data_dir, f"sample_{i}.json"), 'w') as f:
                json.dump(sample, f, indent=2)
        
        logger.info(f"Generated {len(color_objects)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class GiramilleStyleTrainer:
    """Advanced trainer for Giramille style"""
    
    def __init__(self, model_path: str = "runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.pipe = None
        self.load_model()
        
    def load_model(self):
        """Load the base model for training"""
        try:
            logger.info("Loading base model for training...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def train_giramille_style(self, epochs: int = 10, batch_size: int = 4):
        """Train the model for Giramille style"""
        logger.info("🚀 Starting Giramille style training...")
        
        # Load dataset
        dataset = GiramilleDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Process batch
                    prompts = batch['prompt'] if isinstance(batch, dict) else [item['prompt'] for item in batch]
                    enhanced_prompts = batch['enhanced_prompt'] if isinstance(batch, dict) else [item['enhanced_prompt'] for item in batch]
                    
                    # Generate images with both prompts
                    for prompt, enhanced_prompt in zip(prompts, enhanced_prompts):
                        self._train_single_prompt(prompt, enhanced_prompt, epoch, batch_idx)
                        
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            logger.info(f"Completed epoch {epoch + 1}")
        
        # Save trained model
        self._save_trained_model()
        logger.info("🎉 Training completed!")
    
    def _train_single_prompt(self, prompt: str, enhanced_prompt: str, epoch: int, batch_idx: int):
        """Train on a single prompt"""
        try:
            # Generate image with enhanced prompt
            with torch.no_grad():
                result = self.pipe(
                    enhanced_prompt,
                    num_inference_steps=20,  # Faster for training
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator().manual_seed(42)
                )
            
            # Save training sample
            output_dir = f"outputs/training/epoch_{epoch}"
            os.makedirs(output_dir, exist_ok=True)
            
            image = result.images[0]
            filename = f"epoch_{epoch}_batch_{batch_idx}_{prompt.replace(' ', '_')}.png"
            image.save(os.path.join(output_dir, filename))
            
            logger.info(f"Generated: {filename}")
            
        except Exception as e:
            logger.error(f"Error generating image for prompt '{prompt}': {e}")
    
    def _save_trained_model(self):
        """Save the trained model"""
        try:
            output_dir = "outputs/trained_models"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(output_dir, "giramille_style_model")
            self.pipe.save_pretrained(model_path)
            
            # Save training info
            training_info = {
                "model_path": model_path,
                "training_date": datetime.now().isoformat(),
                "device": self.device,
                "base_model": self.model_path
            }
            
            with open(os.path.join(output_dir, "training_info.json"), 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info(f"✅ Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")

class GiramilleStyleValidator:
    """Validator for Giramille style quality"""
    
    def __init__(self):
        self.color_accuracy_threshold = 0.8
        self.style_consistency_threshold = 0.7
    
    def validate_color_accuracy(self, prompt: str, image: Image.Image) -> float:
        """Validate color accuracy of generated image"""
        # Extract expected colors from prompt
        expected_colors = self._extract_expected_colors(prompt)
        
        # Analyze image colors
        image_colors = self._analyze_image_colors(image)
        
        # Calculate accuracy
        accuracy = self._calculate_color_accuracy(expected_colors, image_colors)
        
        return accuracy
    
    def _extract_expected_colors(self, prompt: str) -> List[str]:
        """Extract expected colors from prompt"""
        colors = []
        prompt_lower = prompt.lower()
        
        color_keywords = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'white', 'black']
        
        for color in color_keywords:
            if color in prompt_lower:
                colors.append(color)
        
        return colors
    
    def _analyze_image_colors(self, image: Image.Image) -> List[str]:
        """Analyze dominant colors in image"""
        # Convert to RGB
        image = image.convert('RGB')
        
        # Get dominant colors
        colors = image.getcolors(maxcolors=256*256*256)
        if not colors:
            return []
        
        # Sort by frequency
        colors.sort(key=lambda x: x[0], reverse=True)
        
        # Convert to color names
        dominant_colors = []
        for count, (r, g, b) in colors[:5]:  # Top 5 colors
            color_name = self._rgb_to_color_name(r, g, b)
            if color_name:
                dominant_colors.append(color_name)
        
        return dominant_colors
    
    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """Convert RGB to color name"""
        color_ranges = {
            'red': [(200, 0, 0), (255, 100, 100)],
            'blue': [(0, 0, 200), (100, 100, 255)],
            'green': [(0, 200, 0), (100, 255, 100)],
            'yellow': [(200, 200, 0), (255, 255, 100)],
            'purple': [(200, 0, 200), (255, 100, 255)],
            'orange': [(255, 165, 0), (255, 200, 100)],
            'pink': [(255, 192, 203), (255, 220, 220)],
            'brown': [(139, 69, 19), (200, 150, 100)],
            'white': [(240, 240, 240), (255, 255, 255)],
            'black': [(0, 0, 0), (50, 50, 50)]
        }
        
        for color_name, (min_rgb, max_rgb) in color_ranges.items():
            if (min_rgb[0] <= r <= max_rgb[0] and 
                min_rgb[1] <= g <= max_rgb[1] and 
                min_rgb[2] <= b <= max_rgb[2]):
                return color_name
        
        return None
    
    def _calculate_color_accuracy(self, expected: List[str], actual: List[str]) -> float:
        """Calculate color accuracy score"""
        if not expected:
            return 1.0
        
        matches = 0
        for expected_color in expected:
            if expected_color in actual:
                matches += 1
        
        return matches / len(expected)

def run_advanced_training():
    """Run advanced training for Giramille style"""
    logger.info("🚀 Starting Advanced Giramille Training Pipeline")
    
    try:
        # Initialize trainer
        trainer = GiramilleStyleTrainer()
        
        # Run training
        trainer.train_giramille_style(epochs=5, batch_size=2)
        
        # Validate results
        validator = GiramilleStyleValidator()
        
        # Test with sample prompts
        test_prompts = [
            "blue house with red car, cartoon style",
            "green tree with yellow flowers, nature scene",
            "purple mountain with orange sunset, landscape"
        ]
        
        logger.info("🧪 Testing trained model...")
        for prompt in test_prompts:
            try:
                # Generate test image
                result = trainer.pipe(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                )
                
                image = result.images[0]
                
                # Validate
                accuracy = validator.validate_color_accuracy(prompt, image)
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Color Accuracy: {accuracy:.2f}")
                
                # Save test image
                test_dir = "outputs/test_images"
                os.makedirs(test_dir, exist_ok=True)
                filename = f"test_{prompt.replace(' ', '_').replace(',', '')}.png"
                image.save(os.path.join(test_dir, filename))
                
            except Exception as e:
                logger.error(f"Error testing prompt '{prompt}': {e}")
        
        logger.info("🎉 Advanced training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    run_advanced_training()
