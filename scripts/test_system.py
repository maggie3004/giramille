#!/usr/bin/env python3
"""
Comprehensive system test for AI Image Generation Giramille.
"""

import os
import json
import sys
from pathlib import Path

def test_dataset_integrity():
    """Test dataset extraction and organization."""
    print("=== DATASET INTEGRITY TEST ===")
    
    train_dir = Path('data/train')
    val_dir = Path('data/val')
    
    # Count images
    train_count = len([f for f in train_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']])
    val_count = len([f for f in val_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']])
    
    print(f"✅ Training images: {train_count}")
    print(f"✅ Validation images: {val_count}")
    print(f"✅ Total images: {train_count + val_count}")
    
    # Check if we have the expected number
    expected_total = 320
    actual_total = train_count + val_count
    
    if actual_total == expected_total:
        print(f"✅ Dataset count matches expected: {expected_total}")
    else:
        print(f"❌ Dataset count mismatch: expected {expected_total}, got {actual_total}")
        return False
    
    return True

def test_dataset_manifest():
    """Test dataset manifest file."""
    print("\n=== DATASET MANIFEST TEST ===")
    
    manifest_path = Path('data/dataset_manifest.json')
    if not manifest_path.exists():
        print("❌ Manifest file missing")
        return False
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        print(f"✅ Manifest exists with {manifest['total_images']} total images")
        print(f"✅ Categories: {len(manifest['categories'])} main categories")
        
        # Check if manifest data matches actual counts
        train_dir = Path('data/train')
        val_dir = Path('data/val')
        actual_train = len([f for f in train_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']])
        actual_val = len([f for f in val_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']])
        
        if manifest['train_images'] == actual_train and manifest['val_images'] == actual_val:
            print("✅ Manifest data matches actual file counts")
        else:
            print(f"❌ Manifest data mismatch: manifest({manifest['train_images']}, {manifest['val_images']}) vs actual({actual_train}, {actual_val})")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading manifest: {e}")
        return False

def test_configuration_files():
    """Test configuration files."""
    print("\n=== CONFIGURATION FILES TEST ===")
    
    config_files = [
        'configs/config.yaml',
        'configs/training_config.yaml',
        'requirements.txt',
        'environment.yml'
    ]
    
    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file} exists")
        else:
            print(f"❌ {config_file} missing")
            all_exist = False
    
    return all_exist

def test_python_imports():
    """Test Python module imports."""
    print("\n=== PYTHON IMPORTS TEST ===")
    
    required_modules = [
        'torch',
        'torchvision', 
        'numpy',
        'PIL',
        'cv2',
        'typer',
        'hydra',
        'rich',
        'flask',
        'svgpathtools'
    ]
    
    all_imports_ok = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ {module} import failed: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_frontend_structure():
    """Test frontend structure."""
    print("\n=== FRONTEND STRUCTURE TEST ===")
    
    frontend_files = [
        'frontend/package.json',
        'frontend/app/page.tsx',
        'frontend/app/layout.tsx',
        'frontend/components/History.tsx',
        'frontend/tailwind.config.js',
        'frontend/next.config.js'
    ]
    
    all_exist = True
    for file_path in frontend_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_scripts():
    """Test automation scripts."""
    print("\n=== SCRIPTS TEST ===")
    
    scripts = [
        'scripts/prepare_dataset.py',
        'scripts/setup_training.py',
        'scripts/split_dataset.py',
        'scripts/extract_dataset.py',
        'scripts/offline_guard.py',
        'scripts/dataset_manifest.py'
    ]
    
    all_exist = True
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} missing")
            all_exist = False
    
    return all_exist

def test_model_files():
    """Test model architecture files."""
    print("\n=== MODEL FILES TEST ===")
    
    model_files = [
        'src/models/diffusion_unet.py',
        'src/models/segnet.py',
        'src/data/dataset.py',
        'src/utils/metrics.py',
        'src/vector/curve_fit.py',
        'src/vector/export.py'
    ]
    
    all_exist = True
    for file_path in model_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_directory_structure():
    """Test overall directory structure."""
    print("\n=== DIRECTORY STRUCTURE TEST ===")
    
    required_dirs = [
        'data/train',
        'data/val',
        'data/processed',
        'checkpoints',
        'logs',
        'outputs/generated',
        'outputs/vectors',
        'frontend',
        'backend',
        'src/models',
        'src/data',
        'src/utils',
        'src/vector',
        'scripts',
        'configs'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("🧪 AI IMAGE GENERATION GIRAMILLE - SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        ("Dataset Integrity", test_dataset_integrity),
        ("Dataset Manifest", test_dataset_manifest),
        ("Configuration Files", test_configuration_files),
        ("Python Imports", test_python_imports),
        ("Frontend Structure", test_frontend_structure),
        ("Scripts", test_scripts),
        ("Model Files", test_model_files),
        ("Directory Structure", test_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
