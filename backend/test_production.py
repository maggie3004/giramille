import os
import sys
import io
from production_system import initialize_production_system
from PIL import Image

def test_production_system():
    print("Testing Giramille Production System...")
    
    # Initialize system with CPU configuration
    generator = initialize_production_system(device_type="cpu")
    
    # Test prompt
    test_prompt = "blue house with red roof, cartoon style, Giramille style"
    print(f"\nGenerating test image with prompt: {test_prompt}")
    
    # Generate image
    result = generator.generate_image(test_prompt, quality="balanced")
    
    if result['success']:
        print(f"\nSuccess! Image generated in {result['generation_time']:.2f}s")
        print(f"Cached: {result['cached']}")
        
        # Save the test image
        if isinstance(result['image'], bytes):
            image = Image.open(io.BytesIO(result['image']))
        else:
            image = result['image']
            
        # Create test output directory
        os.makedirs('test_outputs', exist_ok=True)
        output_path = os.path.join('test_outputs', 'test_production_output.png')
        image.save(output_path)
        print(f"\nTest image saved to: {output_path}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_production_system()