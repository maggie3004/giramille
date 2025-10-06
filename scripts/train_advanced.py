#!/usr/bin/env python3
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
