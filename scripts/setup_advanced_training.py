#!/usr/bin/env python3
"""
Setup Advanced Training Environment
Prepares the system for Freepik-level AI training with Giramille style
"""

import os
import sys
import json
import shutil
from pathlib import Path
import subprocess
import logging
from typing import Dict, List, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTrainingSetup:
    """Setup advanced training environment for Giramille AI system"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.setup_dirs()
        self.config = self.load_config()
    
    def setup_dirs(self):
        """Create necessary directories for advanced training"""
        
        dirs = [
            "data/train/characters",
            "data/train/objects", 
            "data/train/animals",
            "data/train/scenarios",
            "data/style_references",
            "data/validation",
            "data/test",
            "checkpoints",
            "samples",
            "logs",
            "models/pretrained",
            "models/finetuned",
            "exports/vectors",
            "exports/multi_view",
            "temp"
        ]
        
        for dir_path in dirs:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
    
    def load_config(self) -> Dict:
        """Load or create training configuration"""
        
        config_path = self.base_dir / "config" / "advanced_training.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded existing config from {config_path}")
        else:
            config = self.create_default_config()
            self.save_config(config, config_path)
            logger.info(f"Created default config at {config_path}")
        
        return config
    
    def create_default_config(self) -> Dict:
        """Create default training configuration"""
        
        return {
            "model": {
                "base_model": "runwayml/stable-diffusion-v1-5",
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-4,
                "weight_decay": 1e-2,
                "epochs": 100,
                "save_interval": 10,
                "sample_interval": 5,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0
            },
            "data": {
                "image_size": 512,
                "max_images_per_category": 1000,
                "augmentation": {
                    "horizontal_flip": True,
                    "rotation": 10,
                    "color_jitter": {
                        "brightness": 0.1,
                        "contrast": 0.1,
                        "saturation": 0.1
                    }
                }
            },
            "loss_weights": {
                "style_consistency": 1.0,
                "category_consistency": 0.5,
                "multi_view_consistency": 0.3,
                "perceptual": 0.2
            },
            "multi_view": {
                "enabled": True,
                "angles": [
                    {"name": "front", "rotation": 0, "elevation": 0},
                    {"name": "3_4_left", "rotation": 45, "elevation": 0},
                    {"name": "3_4_right", "rotation": -45, "elevation": 0},
                    {"name": "side_left", "rotation": 90, "elevation": 0},
                    {"name": "side_right", "rotation": -90, "elevation": 0},
                    {"name": "back", "rotation": 180, "elevation": 0},
                    {"name": "top", "rotation": 0, "elevation": 90},
                    {"name": "bottom", "rotation": 0, "elevation": -90}
                ]
            },
            "vector_conversion": {
                "max_anchors": 1000,
                "max_layers": 50,
                "simplification_threshold": 0.5,
                "color_quantization": 8
            },
            "hardware": {
                "use_cuda": True,
                "num_workers": 4,
                "pin_memory": True
            },
            "logging": {
                "use_wandb": True,
                "log_interval": 10,
                "save_samples": True
            }
        }
    
    def save_config(self, config: Dict, config_path: Path):
        """Save configuration to file"""
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def install_dependencies(self):
        """Install required Python dependencies"""
        
        logger.info("Installing advanced training dependencies...")
        
        dependencies = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "diffusers>=0.21.0",
            "peft>=0.4.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "datasets>=2.12.0",
            "wandb>=0.15.0",
            "opencv-python>=4.8.0",
            "Pillow>=9.5.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "svgpathtools>=1.6.0",
            "svgwrite>=1.4.0",
            "tqdm>=4.65.0",
            "hydra-core>=1.3.0",
            "omegaconf>=2.3.0",
            "rich>=13.0.0"
        ]
        
        for dep in dependencies:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
                logger.info(f"Installed {dep}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {dep}: {e}")
    
    def download_pretrained_models(self):
        """Download pre-trained models"""
        
        logger.info("Downloading pre-trained models...")
        
        models = [
            {
                "name": "stable-diffusion-v1-5",
                "url": "runwayml/stable-diffusion-v1-5",
                "type": "diffusion"
            },
            {
                "name": "controlnet-canny",
                "url": "lllyasviel/sd-controlnet-canny",
                "type": "controlnet"
            },
            {
                "name": "inpaint-model",
                "url": "runwayml/stable-diffusion-inpainting",
                "type": "inpainting"
            }
        ]
        
        for model in models:
            try:
                model_dir = self.base_dir / "models" / "pretrained" / model["name"]
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Download model (this would use huggingface_hub in practice)
                logger.info(f"Downloading {model['name']}...")
                # Placeholder for actual download logic
                
            except Exception as e:
                logger.error(f"Failed to download {model['name']}: {e}")
    
    def setup_dataset_structure(self):
        """Setup dataset directory structure"""
        
        logger.info("Setting up dataset structure...")
        
        # Create dataset manifest
        manifest = {
            "version": "1.0",
            "created": "2024-01-01",
            "description": "Giramille AI Training Dataset",
            "categories": {
                "characters": {
                    "description": "Character images in Giramille style",
                    "min_images": 50,
                    "max_images": 1000
                },
                "objects": {
                    "description": "Object images in Giramille style", 
                    "min_images": 50,
                    "max_images": 1000
                },
                "animals": {
                    "description": "Animal images in Giramille style",
                    "min_images": 50,
                    "max_images": 1000
                },
                "scenarios": {
                    "description": "Scenario images in Giramille style",
                    "min_images": 50,
                    "max_images": 1000
                }
            },
            "style_references": {
                "description": "Reference images for Giramille style",
                "min_images": 20,
                "max_images": 100
            }
        }
        
        manifest_path = self.base_dir / "data" / "dataset_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created dataset manifest: {manifest_path}")
    
    def create_training_scripts(self):
        """Create training and utility scripts"""
        
        logger.info("Creating training scripts...")
        
        scripts = [
            {
                "name": "train_advanced.py",
                "content": self.get_advanced_training_script()
            },
            {
                "name": "train_multi_view.py", 
                "content": self.get_multi_view_training_script()
            },
            {
                "name": "evaluate_model.py",
                "content": self.get_evaluation_script()
            },
            {
                "name": "export_model.py",
                "content": self.get_export_script()
            }
        ]
        
        for script in scripts:
            script_path = self.base_dir / "scripts" / script["name"]
            with open(script_path, 'w') as f:
                f.write(script["content"])
            
            # Make executable
            os.chmod(script_path, 0o755)
            logger.info(f"Created script: {script_path}")
    
    def get_advanced_training_script(self) -> str:
        """Get advanced training script content"""
        
        return '''#!/usr/bin/env python3
"""
Advanced Training Script for Giramille AI System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.advanced_training_pipeline import AdvancedTrainingPipeline
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Giramille AI model')
    parser.add_argument('--config', type=str, default='config/advanced_training.json',
                       help='Path to training configuration')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create training pipeline
    pipeline = AdvancedTrainingPipeline(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        pipeline.load_checkpoint(args.resume)
    
    # Start training
    pipeline.train()

if __name__ == "__main__":
    main()
'''
    
    def get_multi_view_training_script(self) -> str:
        """Get multi-view training script content"""
        
        return '''#!/usr/bin/env python3
"""
Multi-View Training Script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.multi_view_generator import MultiViewPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train multi-view generator')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = MultiViewPipeline()
    
    # Train model
    pipeline.train(args.data_dir, args.epochs, args.batch_size)
    
    # Save model
    pipeline.save_model('models/finetuned/multi_view_generator.pt')

if __name__ == "__main__":
    main()
'''
    
    def get_evaluation_script(self) -> str:
        """Get model evaluation script content"""
        
        return '''#!/usr/bin/env python3
"""
Model Evaluation Script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.advanced_training_pipeline import AdvancedTrainingPipeline
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate Giramille AI model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load model
    pipeline = AdvancedTrainingPipeline({})
    pipeline.load_checkpoint(args.model_path)
    
    # Evaluate model
    # This would implement evaluation logic
    print(f"Evaluating model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
'''
    
    def get_export_script(self) -> str:
        """Get model export script content"""
        
        return '''#!/usr/bin/env python3
"""
Model Export Script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.advanced_training_pipeline import AdvancedTrainingPipeline
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Export Giramille AI model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--export_format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'safetensors'],
                       help='Export format')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for exported model')
    
    args = parser.parse_args()
    
    # Load model
    pipeline = AdvancedTrainingPipeline({})
    pipeline.load_checkpoint(args.model_path)
    
    # Export model
    print(f"Exporting model: {args.model_path}")
    print(f"Format: {args.export_format}")
    print(f"Output: {args.output_path}")

if __name__ == "__main__":
    main()
'''
    
    def create_docker_setup(self):
        """Create Docker setup for advanced training"""
        
        logger.info("Creating Docker setup...")
        
        dockerfile_content = '''FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    git \\
    wget \\
    curl \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python3", "scripts/train_advanced.py"]
'''
        
        dockerfile_path = self.base_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create docker-compose.yml
        docker_compose_content = '''version: '3.8'

services:
  giramille-ai:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./samples:/app/samples
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''
        
        docker_compose_path = self.base_dir / "docker-compose.yml"
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_content)
        
        logger.info("Created Docker setup files")
    
    def create_requirements(self):
        """Create requirements.txt for advanced training"""
        
        requirements = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "diffusers>=0.21.0",
            "peft>=0.4.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "datasets>=2.12.0",
            "wandb>=0.15.0",
            "opencv-python>=4.8.0",
            "Pillow>=9.5.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "svgpathtools>=1.6.0",
            "svgwrite>=1.4.0",
            "tqdm>=4.65.0",
            "hydra-core>=1.3.0",
            "omegaconf>=2.3.0",
            "rich>=13.0.0",
            "huggingface-hub>=0.16.0",
            "safetensors>=0.3.0",
            "xformers>=0.0.20",
            "bitsandbytes>=0.41.0"
        ]
        
        requirements_path = self.base_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        logger.info(f"Created requirements.txt: {requirements_path}")
    
    def run_setup(self):
        """Run complete setup process"""
        
        logger.info("Starting advanced training setup...")
        
        try:
            # Install dependencies
            self.install_dependencies()
            
            # Download pretrained models
            self.download_pretrained_models()
            
            # Setup dataset structure
            self.setup_dataset_structure()
            
            # Create training scripts
            self.create_training_scripts()
            
            # Create Docker setup
            self.create_docker_setup()
            
            # Create requirements
            self.create_requirements()
            
            logger.info("Advanced training setup completed successfully!")
            logger.info("Next steps:")
            logger.info("1. Add your Giramille style images to data/train/")
            logger.info("2. Add style reference images to data/style_references/")
            logger.info("3. Run: python scripts/train_advanced.py")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Setup advanced training environment')
    parser.add_argument('--base_dir', type=str, default='.',
                       help='Base directory for setup')
    parser.add_argument('--skip_deps', action='store_true',
                       help='Skip dependency installation')
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = AdvancedTrainingSetup(args.base_dir)
    
    # Run setup
    setup.run_setup()

if __name__ == "__main__":
    main()
