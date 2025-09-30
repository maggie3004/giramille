#!/usr/bin/env python3
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
