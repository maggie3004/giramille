#!/usr/bin/env python3
"""
Safe dataset extraction script that handles long paths and security issues.
"""

import zipfile
import os
import shutil
from pathlib import Path
import re

def safe_extract_zip(zip_path, extract_to):
    """Extract zip file safely, handling long paths and security issues."""
    
    # Create extraction directory
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = extract_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files
        file_list = zip_ref.namelist()
        print(f"Found {len(file_list)} files in zip")
        
        # Filter and extract only image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        extracted_count = 0
        
        for file_info in zip_ref.filelist:
            # Skip directories
            if file_info.is_dir():
                continue
                
            # Get filename
            filename = os.path.basename(file_info.filename)
            
            # Skip if no filename or hidden files
            if not filename or filename.startswith('.'):
                continue
                
            # Check if it's an image file
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in image_extensions:
                continue
                
            # Create safe filename (remove special characters)
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            safe_filename = safe_filename[:100]  # Limit length
            
            # Skip if filename is too long or empty
            if not safe_filename or len(safe_filename) < 3:
                continue
                
            try:
                # Extract to images directory
                target_path = images_dir / safe_filename
                
                # If file already exists, add number suffix
                counter = 1
                original_target = target_path
                while target_path.exists():
                    name, ext = os.path.splitext(original_target.name)
                    target_path = images_dir / f"{name}_{counter}{ext}"
                    counter += 1
                
                # Extract file
                with zip_ref.open(file_info) as source:
                    with open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                
                extracted_count += 1
                if extracted_count % 10 == 0:
                    print(f"Extracted {extracted_count} images...")
                    
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                continue
    
    print(f"Successfully extracted {extracted_count} images to {images_dir}")
    return extracted_count

if __name__ == "__main__":
    import sys
    
    zip_file = "7. Banco de Imagens.zip"
    extract_dir = "sample_dataset"
    
    if len(sys.argv) > 1:
        zip_file = sys.argv[1]
    if len(sys.argv) > 2:
        extract_dir = sys.argv[2]
    
    if not os.path.exists(zip_file):
        print(f"Error: Zip file {zip_file} not found")
        sys.exit(1)
    
    count = safe_extract_zip(zip_file, extract_dir)
    print(f"Extraction complete! {count} images extracted.")
