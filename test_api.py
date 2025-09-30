<<<<<<< HEAD
import requests
import json

def test_api():
    try:
        # Test health endpoint
        health_response = requests.get('http://localhost:5000/api/health')
        print("Health Check:", health_response.json())
        
        # Test generation endpoint
        generation_data = {
            "prompt": "blue house with red car",
            "style": "png"
        }
        
        response = requests.post(
            'http://localhost:5000/api/generate',
            headers={'Content-Type': 'application/json'},
            json=generation_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Prompt: {result['prompt']}")
            print(f"Style: {result['style']}")
            print(f"Image size: {len(result['image'])} characters")
            print(f"Generated at: {result['generated_at']}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_api()
=======
import requests
import json

def test_api():
    try:
        # Test health endpoint
        health_response = requests.get('http://localhost:5000/api/health')
        print("Health Check:", health_response.json())
        
        # Test generation endpoint
        generation_data = {
            "prompt": "blue house with red car",
            "style": "png"
        }
        
        response = requests.post(
            'http://localhost:5000/api/generate',
            headers={'Content-Type': 'application/json'},
            json=generation_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Prompt: {result['prompt']}")
            print(f"Style: {result['style']}")
            print(f"Image size: {len(result['image'])} characters")
            print(f"Generated at: {result['generated_at']}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_api()
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
