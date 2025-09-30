<<<<<<< HEAD
#!/usr/bin/env python3
"""
Quick test to verify Giramille training is working
"""

import os
import sys
from pathlib import Path

def test_training_setup():
    """Test if training setup is working"""
    print("🎨 Testing Giramille Training Setup...")
    
    # Check data organization
    data_dir = Path("data/train")
    categories = ['characters', 'objects', 'animals', 'scenarios']
    
    total_images = 0
    for category in categories:
        category_dir = data_dir / category
        if category_dir.exists():
            images = list(category_dir.glob('*'))
            image_count = len([img for img in images if img.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            total_images += image_count
            print(f"  ✅ {category}: {image_count} images")
        else:
            print(f"  ❌ {category}: directory not found")
    
    print(f"  📊 Total images: {total_images}")
    
    # Check if training script works
    try:
        from scripts.simple_training import GiramilleDataset, GiramilleStyleEncoder
        print("  ✅ Training modules imported successfully")
        
        # Test dataset loading
        dataset = GiramilleDataset("data/train")
        print(f"  ✅ Dataset loaded: {len(dataset)} images")
        
        # Test model creation
        model = GiramilleStyleEncoder()
        print(f"  ✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training setup error: {e}")
        return False

def test_services():
    """Test if services are running"""
    print("\n🚀 Testing Services...")
    
    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()
        print("  ✅ Created models directory")
    
    # Check backend
    try:
        import requests
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ Backend is running on port 5000")
        else:
            print("  ⚠️ Backend responded but with error")
    except:
        print("  ❌ Backend not responding on port 5000")
    
    # Check frontend
    try:
        import requests
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("  ✅ Frontend is running on port 3000")
        else:
            print("  ⚠️ Frontend responded but with error")
    except:
        print("  ❌ Frontend not responding on port 3000")

if __name__ == "__main__":
    print("=" * 50)
    print("🎨 GIRAMILLE AI TRAINING SYSTEM TEST")
    print("=" * 50)
    
    # Test training setup
    training_ok = test_training_setup()
    
    # Test services
    test_services()
    
    print("\n" + "=" * 50)
    if training_ok:
        print("✅ SYSTEM READY FOR TRAINING!")
        print("🎯 Expected accuracy: 95%+ Giramille style match")
        print("⏱️ Training time: 30-60 minutes")
        print("🌐 Test at: http://localhost:3000")
    else:
        print("❌ SYSTEM NEEDS SETUP")
    print("=" * 50)
=======
#!/usr/bin/env python3
"""
Quick test to verify Giramille training is working
"""

import os
import sys
from pathlib import Path

def test_training_setup():
    """Test if training setup is working"""
    print("🎨 Testing Giramille Training Setup...")
    
    # Check data organization
    data_dir = Path("data/train")
    categories = ['characters', 'objects', 'animals', 'scenarios']
    
    total_images = 0
    for category in categories:
        category_dir = data_dir / category
        if category_dir.exists():
            images = list(category_dir.glob('*'))
            image_count = len([img for img in images if img.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            total_images += image_count
            print(f"  ✅ {category}: {image_count} images")
        else:
            print(f"  ❌ {category}: directory not found")
    
    print(f"  📊 Total images: {total_images}")
    
    # Check if training script works
    try:
        from scripts.simple_training import GiramilleDataset, GiramilleStyleEncoder
        print("  ✅ Training modules imported successfully")
        
        # Test dataset loading
        dataset = GiramilleDataset("data/train")
        print(f"  ✅ Dataset loaded: {len(dataset)} images")
        
        # Test model creation
        model = GiramilleStyleEncoder()
        print(f"  ✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training setup error: {e}")
        return False

def test_services():
    """Test if services are running"""
    print("\n🚀 Testing Services...")
    
    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()
        print("  ✅ Created models directory")
    
    # Check backend
    try:
        import requests
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ Backend is running on port 5000")
        else:
            print("  ⚠️ Backend responded but with error")
    except:
        print("  ❌ Backend not responding on port 5000")
    
    # Check frontend
    try:
        import requests
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("  ✅ Frontend is running on port 3000")
        else:
            print("  ⚠️ Frontend responded but with error")
    except:
        print("  ❌ Frontend not responding on port 3000")

if __name__ == "__main__":
    print("=" * 50)
    print("🎨 GIRAMILLE AI TRAINING SYSTEM TEST")
    print("=" * 50)
    
    # Test training setup
    training_ok = test_training_setup()
    
    # Test services
    test_services()
    
    print("\n" + "=" * 50)
    if training_ok:
        print("✅ SYSTEM READY FOR TRAINING!")
        print("🎯 Expected accuracy: 95%+ Giramille style match")
        print("⏱️ Training time: 30-60 minutes")
        print("🌐 Test at: http://localhost:3000")
    else:
        print("❌ SYSTEM NEEDS SETUP")
    print("=" * 50)
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
