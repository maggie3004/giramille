"""
Generate 3 Proof-of-Concept Images for Giramille Style
This will create high-quality images like Gemini/Freepik
"""

import requests
import json
import base64
from PIL import Image
import io
import os

def generate_proof_concept_images():
    """Generate 3 proof-of-concept images"""
    
    # Test prompts for different scenarios
    prompts = [
        "blue house with red car in front, sunny day, cartoon style",
        "green tree with yellow flowers, nature scene, colorful",
        "purple mountain landscape with orange sunset, beautiful view"
    ]
    
    # API endpoint
    api_url = "http://localhost:5000/api/generate"
    
    print("[START] Generating 3 Proof-of-Concept Images...")
    print("=" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[GENERATING] Image {i}: {prompt}")
        
        try:
            # Make API request
            response = requests.post(api_url, json={
                'prompt': prompt,
                'style': 'png'
            })
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    # Decode base64 image
                    image_data = data['image'].split(',')[1]  # Remove data:image/png;base64, prefix
                    image_bytes = base64.b64decode(image_data)
                    
                    # Save image
                    image = Image.open(io.BytesIO(image_bytes))
                    filename = f"proof_concept_{i}.png"
                    image.save(filename)
                    
                    print(f"[SUCCESS] Image {i} saved as: {filename}")
                    print(f"   Size: {image.size}")
                    print(f"   Generated at: {data.get('generated_at', 'N/A')}")
                    
                else:
                    print(f"[ERROR] Error generating image {i}: {data.get('error', 'Unknown error')}")
            else:
                print(f"[ERROR] API Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[ERROR] Exception generating image {i}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("[COMPLETE] Proof-of-concept generation complete!")
    print("Check the generated PNG files in the current directory.")

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get("http://localhost:5000/api/health")
        if response.status_code == 200:
            print("[SUCCESS] API is running and healthy")
            return True
        else:
            print(f"[ERROR] API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Cannot connect to API: {str(e)}")
        return False

if __name__ == "__main__":
    print("[SYSTEM] Giramille Style Proof-of-Concept Generator")
    print("=" * 50)
    
    # Test API health first
    if test_api_health():
        generate_proof_concept_images()
    else:
        print("\n[INFO] Make sure to start the backend server first:")
        print("   python backend/app.py")
