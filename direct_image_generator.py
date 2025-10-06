"""
Direct Image Generator for Giramille Style
Generate 3 proof-of-concept images directly without API
"""

import sys
import os
sys.path.append('backend')

from advanced_generator import generate_giramille_image_advanced
from PIL import Image
import io

def generate_direct_images():
    """Generate 3 proof-of-concept images directly"""
    
    # Test prompts for different scenarios
    prompts = [
        "blue house with red car in front, sunny day, cartoon style",
        "green tree with yellow flowers, nature scene, colorful",
        "purple mountain landscape with orange sunset, beautiful view"
    ]
    
    print("[START] Generating 3 Proof-of-Concept Images Directly...")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[GENERATING] Image {i}: {prompt}")
        
        try:
            # Generate image directly
            image = generate_giramille_image_advanced(prompt, "png")
            
            # Save image
            filename = f"proof_concept_{i}.png"
            image.save(filename)
            
            print(f"[SUCCESS] Image {i} saved as: {filename}")
            print(f"   Size: {image.size}")
            print(f"   Mode: {image.mode}")
            
            # Show some image info
            if hasattr(image, 'info'):
                print(f"   Info: {image.info}")
            
        except Exception as e:
            print(f"[ERROR] Error generating image {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("[COMPLETE] Direct proof-of-concept generation complete!")
    print("Check the generated PNG files in the current directory.")

if __name__ == "__main__":
    print("[SYSTEM] Direct Giramille Style Image Generator")
    print("=" * 60)
    generate_direct_images()
