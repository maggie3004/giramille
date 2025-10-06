"""
Generate High-Quality Proof-of-Concept Images
Using the same quality settings as training for consistent results
"""

import sys
import os
sys.path.append('backend')

from production_system import generate_production_image, initialize_production_system
from PIL import Image
import io
import time

def generate_high_quality_proof_concept():
    """Generate 3 high-quality proof-of-concept images matching training quality"""
    
    # Initialize production system
    print("[START] Initializing High-Quality Production System...")
    generator = initialize_production_system()
    
    # Test prompts for different scenarios (same as before but with high quality)
    prompts = [
        "blue house with red car in front, sunny day, cartoon style",
        "green tree with yellow flowers, nature scene, colorful",
        "purple mountain landscape with orange sunset, beautiful view"
    ]
    
    print("[GENERATING] High-Quality Proof-of-Concept Images...")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[GENERATING] High-Quality Image {i}: {prompt}")
        
        try:
            # Generate with high quality settings
            result = generator.generate_image(prompt, "png", "high")
            
            if result['success']:
                # Save image
                filename = f"high_quality_proof_{i}.png"
                with open(filename, 'wb') as f:
                    f.write(result['image'])
                
                print(f"[SUCCESS] High-Quality Image {i} saved as: {filename}")
                print(f"   Size: 512x512")
                print(f"   Quality: High (50 steps, 7.5 guidance)")
                print(f"   Generation Time: {result['generation_time']:.2f}s")
                print(f"   Cached: {result['cached']}")
                
            else:
                print(f"[ERROR] Error generating image {i}: {result['error']}")
                
        except Exception as e:
            print(f"[ERROR] Exception generating image {i}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("[COMPLETE] High-Quality proof-of-concept generation complete!")
    print("Check the generated PNG files in the current directory.")
    
    # Show system metrics
    try:
        metrics = generator.get_metrics()
        print(f"\n[METRICS] System Metrics:")
        print(f"   Total Requests: {metrics['requests_total']}")
        print(f"   Success Rate: {metrics['requests_success']}/{metrics['requests_total']}")
        print(f"   Average Generation Time: {metrics['avg_generation_time']:.2f}s")
        print(f"   Cache Hit Rate: {metrics['cache_hits']}/{metrics['cache_hits'] + metrics['cache_misses']}")
    except Exception as e:
        print(f"[WARNING] Could not retrieve metrics: {e}")

def compare_quality():
    """Compare different quality settings"""
    print("\n🔍 Quality Comparison Test...")
    
    generator = initialize_production_system()
    test_prompt = "blue house with red car, cartoon style"
    
    qualities = ['fast', 'balanced', 'high']
    
    for quality in qualities:
        print(f"\n[TESTING] {quality} quality...")
        
        try:
            start_time = time.time()
            result = generator.generate_image(test_prompt, "png", quality)
            end_time = time.time()
            
            if result['success']:
                filename = f"quality_comparison_{quality}.png"
                with open(filename, 'wb') as f:
                    f.write(result['image'])
                
                print(f"[SUCCESS] {quality.capitalize()} quality: {end_time - start_time:.2f}s")
                print(f"   Saved as: {filename}")
            else:
                print(f"[ERROR] {quality.capitalize()} quality failed: {result['error']}")
                
        except Exception as e:
            print(f"[ERROR] Exception with {quality} quality: {e}")

if __name__ == "__main__":
    print("[SYSTEM] High-Quality Giramille Proof-of-Concept Generator")
    print("=" * 60)
    
    # Generate high-quality proof-of-concept images
    generate_high_quality_proof_concept()
    
    # Optional: Compare quality settings
    print("\n" + "=" * 60)
    compare_quality()
    
    print("\n[COMPLETE] All tests completed!")
