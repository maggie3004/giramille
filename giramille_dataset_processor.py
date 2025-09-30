<<<<<<< HEAD
"""
Giramille Dataset Processor
Extract, analyze, and prepare the 13GB Giramille image dataset for training
"""

import os
import zipfile
import json
import shutil
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import hashlib
from pathlib import Path
import cv2
from collections import Counter
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GiramilleDatasetProcessor:
    """Process the Giramille image dataset for training"""
    
    def __init__(self, dataset_path: str = "7. Banco de Imagens.zip", output_dir: str = "data/giramille_processed"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.processed_images = []
        self.analysis_results = {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/metadata", exist_ok=True)
        os.makedirs(f"{output_dir}/analysis", exist_ok=True)
    
    def extract_dataset(self) -> bool:
        """Extract the Giramille dataset from zip file"""
        try:
            logger.info(f"🔓 Extracting Giramille dataset from {self.dataset_path}...")
            
            if not os.path.exists(self.dataset_path):
                logger.error(f"❌ Dataset file not found: {self.dataset_path}")
                return False
            
            # Extract to temporary directory first
            temp_dir = "temp/giramille_extraction"
            os.makedirs(temp_dir, exist_ok=True)
            
            with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            logger.info("✅ Dataset extracted successfully!")
            
            # Move to processed directory
            self._organize_extracted_files(temp_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error extracting dataset: {e}")
            return False
    
    def _organize_extracted_files(self, temp_dir: str):
        """Organize extracted files by type and category"""
        logger.info("📁 Organizing extracted files...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        total_images = 0
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                if file_ext in image_extensions:
                    # Copy image to organized structure
                    relative_path = os.path.relpath(file_path, temp_dir)
                    new_path = os.path.join(self.output_dir, "images", relative_path)
                    
                    # Create directory if needed
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(file_path, new_path)
                    total_images += 1
                    
                    # Process image metadata
                    self._process_image_metadata(file_path, new_path)
        
        logger.info(f"✅ Organized {total_images} images")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
    
    def _process_image_metadata(self, original_path: str, new_path: str):
        """Process and store image metadata"""
        try:
            # Load image
            image = Image.open(original_path)
            
            # Basic metadata
            metadata = {
                'original_path': original_path,
                'new_path': new_path,
                'filename': os.path.basename(new_path),
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'file_size': os.path.getsize(original_path),
                'hash': self._calculate_image_hash(original_path),
                'processed_at': datetime.now().isoformat()
            }
            
            # Analyze image content
            analysis = self._analyze_image_content(image)
            metadata.update(analysis)
            
            self.processed_images.append(metadata)
            
        except Exception as e:
            logger.error(f"Error processing {original_path}: {e}")
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash for image deduplication"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _analyze_image_content(self, image: Image.Image) -> Dict:
        """Analyze image content for training categorization"""
        try:
            # Convert to RGB for analysis
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for analysis
            image_small = image.resize((224, 224))
            img_array = np.array(image_small)
            
            # Color analysis
            dominant_colors = self._get_dominant_colors(img_array)
            
            # Brightness and contrast
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Detect if it's likely a Giramille style image
            is_giramille_style = self._detect_giramille_style(img_array)
            
            return {
                'dominant_colors': dominant_colors,
                'brightness': float(brightness),
                'contrast': float(contrast),
                'is_giramille_style': is_giramille_style,
                'color_diversity': len(set(tuple(rgb) for rgb in dominant_colors))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image content: {e}")
            return {}
    
    def _get_dominant_colors(self, img_array: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Get dominant colors using K-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape image array
            pixels = img_array.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            return [tuple(color) for color in colors]
            
        except ImportError:
            # Fallback to simple color analysis
            return self._simple_color_analysis(img_array)
        except Exception as e:
            logger.error(f"Error in color analysis: {e}")
            return []
    
    def _simple_color_analysis(self, img_array: np.ndarray) -> List[Tuple[int, int, int]]:
        """Simple color analysis fallback"""
        # Get unique colors and their frequencies
        pixels = img_array.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        # Return top 5 colors
        top_colors = unique_colors[sorted_indices[:5]]
        return [tuple(color) for color in top_colors]
    
    def _detect_giramille_style(self, img_array: np.ndarray) -> bool:
        """Detect if image matches Giramille style characteristics"""
        try:
            # Giramille style characteristics:
            # - Bright, vibrant colors
            # - High contrast
            # - Cartoon-like appearance
            # - Clean, flat design elements
            
            # Calculate brightness and saturation
            brightness = np.mean(img_array)
            saturation = np.std(img_array)
            
            # Check for vibrant colors (high saturation)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation_values = hsv[:, :, 1]
            avg_saturation = np.mean(saturation_values)
            
            # Giramille style indicators
            is_bright = brightness > 150  # Bright images
            is_saturated = avg_saturation > 100  # Vibrant colors
            is_high_contrast = saturation > 80  # High contrast
            
            # Simple heuristic for Giramille style
            giramille_score = sum([is_bright, is_saturated, is_high_contrast])
            
            return giramille_score >= 2
            
        except Exception as e:
            logger.error(f"Error detecting Giramille style: {e}")
            return False
    
    def analyze_dataset(self) -> Dict:
        """Analyze the processed dataset"""
        logger.info("📊 Analyzing Giramille dataset...")
        
        if not self.processed_images:
            logger.error("No processed images found. Run extract_dataset() first.")
            return {}
        
        # Basic statistics
        total_images = len(self.processed_images)
        total_size = sum(img['file_size'] for img in self.processed_images)
        
        # Size distribution
        sizes = [img['size'] for img in self.processed_images]
        width_dist = [size[0] for size in sizes]
        height_dist = [size[1] for size in sizes]
        
        # Format distribution
        formats = [img['format'] for img in self.processed_images]
        format_counts = Counter(formats)
        
        # Giramille style detection
        giramille_style_count = sum(1 for img in self.processed_images if img.get('is_giramille_style', False))
        
        # Color analysis
        all_colors = []
        for img in self.processed_images:
            all_colors.extend(img.get('dominant_colors', []))
        
        # Brightness and contrast analysis
        brightness_values = [img.get('brightness', 0) for img in self.processed_images]
        contrast_values = [img.get('contrast', 0) for img in self.processed_images]
        
        analysis = {
            'total_images': total_images,
            'total_size_mb': total_size / (1024 * 1024),
            'size_distribution': {
                'widths': {'min': min(width_dist), 'max': max(width_dist), 'avg': np.mean(width_dist)},
                'heights': {'min': min(height_dist), 'max': max(height_dist), 'avg': np.mean(height_dist)}
            },
            'format_distribution': dict(format_counts),
            'giramille_style_detection': {
                'total_giramille_style': giramille_style_count,
                'percentage': (giramille_style_count / total_images) * 100
            },
            'color_analysis': {
                'total_unique_colors': len(set(all_colors)),
                'avg_brightness': np.mean(brightness_values),
                'avg_contrast': np.mean(contrast_values)
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        self.analysis_results = analysis
        
        # Save analysis results
        with open(f"{self.output_dir}/analysis/dataset_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info("✅ Dataset analysis completed!")
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _print_analysis_summary(self, analysis: Dict):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("📊 GIRAMILLE DATASET ANALYSIS SUMMARY")
        print("="*60)
        print(f"📁 Total Images: {analysis['total_images']:,}")
        print(f"💾 Total Size: {analysis['total_size_mb']:.1f} MB")
        print(f"🎨 Giramille Style Images: {analysis['giramille_style_detection']['total_giramille_style']:,} ({analysis['giramille_style_detection']['percentage']:.1f}%)")
        print(f"🌈 Unique Colors: {analysis['color_analysis']['total_unique_colors']:,}")
        print(f"💡 Average Brightness: {analysis['color_analysis']['avg_brightness']:.1f}")
        print(f"⚡ Average Contrast: {analysis['color_analysis']['avg_contrast']:.1f}")
        print("\n📋 Format Distribution:")
        for format_type, count in analysis['format_distribution'].items():
            print(f"   {format_type}: {count:,} images")
        print("="*60)
    
    def create_training_dataset(self) -> bool:
        """Create training dataset from processed images"""
        logger.info("🎯 Creating training dataset...")
        
        try:
            # Filter Giramille style images
            giramille_images = [img for img in self.processed_images if img.get('is_giramille_style', False)]
            
            if not giramille_images:
                logger.warning("No Giramille style images detected. Using all images.")
                giramille_images = self.processed_images
            
            # Create training pairs
            training_pairs = []
            
            for img in giramille_images:
                # Generate prompt based on image analysis
                prompt = self._generate_prompt_from_image(img)
                
                training_pair = {
                    'image_path': img['new_path'],
                    'prompt': prompt,
                    'metadata': img,
                    'training_ready': True
                }
                
                training_pairs.append(training_pair)
            
            # Save training dataset
            training_file = f"{self.output_dir}/metadata/training_dataset.json"
            with open(training_file, 'w') as f:
                json.dump(training_pairs, f, indent=2)
            
            logger.info(f"✅ Created training dataset with {len(training_pairs)} pairs")
            logger.info(f"💾 Saved to: {training_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating training dataset: {e}")
            return False
    
    def _generate_prompt_from_image(self, img_metadata: Dict) -> str:
        """Generate training prompt from image metadata"""
        # Extract dominant colors
        colors = img_metadata.get('dominant_colors', [])
        color_names = [self._rgb_to_color_name(r, g, b) for r, g, b in colors[:3]]
        color_text = ", ".join([c for c in color_names if c])
        
        # Extract filename for context
        filename = img_metadata.get('filename', '').lower()
        
        # Generate base prompt
        base_prompt = "Giramille style, cartoon, colorful, vibrant"
        
        if color_text:
            base_prompt += f", {color_text}"
        
        # Add style indicators based on filename
        if 'casa' in filename or 'house' in filename:
            base_prompt += ", house, building"
        elif 'carro' in filename or 'car' in filename:
            base_prompt += ", car, vehicle"
        elif 'arvore' in filename or 'tree' in filename:
            base_prompt += ", tree, nature"
        elif 'personagem' in filename or 'character' in filename:
            base_prompt += ", character, person"
        
        return base_prompt
    
    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """Convert RGB to color name"""
        color_ranges = {
            'red': [(150, 0, 0), (255, 150, 150)],
            'blue': [(0, 0, 150), (150, 150, 255)],
            'green': [(0, 150, 0), (150, 255, 150)],
            'yellow': [(150, 150, 0), (255, 255, 150)],
            'purple': [(150, 0, 150), (255, 150, 255)],
            'orange': [(255, 165, 0), (255, 200, 100)],
            'pink': [(255, 192, 203), (255, 220, 220)],
            'brown': [(139, 69, 19), (200, 150, 100)],
            'white': [(200, 200, 200), (255, 255, 255)],
            'black': [(0, 0, 0), (100, 100, 100)]
        }
        
        for color_name, (min_rgb, max_rgb) in color_ranges.items():
            if (min_rgb[0] <= r <= max_rgb[0] and 
                min_rgb[1] <= g <= max_rgb[1] and 
                min_rgb[2] <= b <= max_rgb[2]):
                return color_name
        
        return None
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        report = f"""
# GIRAMILLE DATASET PROCESSING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Total Images Processed:** {len(self.processed_images):,}
- **Giramille Style Images:** {sum(1 for img in self.processed_images if img.get('is_giramille_style', False)):,}
- **Total Dataset Size:** {sum(img['file_size'] for img in self.processed_images) / (1024*1024):.1f} MB

## Training Readiness
- **Dataset Status:** {'✅ Ready' if self.analysis_results else '❌ Needs Processing'}
- **Training Pairs Created:** {len([img for img in self.processed_images if img.get('is_giramille_style', False)]):,}
- **Quality Assessment:** {'High' if self.analysis_results.get('giramille_style_detection', {}).get('percentage', 0) > 50 else 'Medium'}

## Next Steps
1. **Cloud Training Setup:** Recommended for full dataset training
2. **Fine-tuning:** Use processed Giramille images for style-specific training
3. **Validation:** Test trained model against Giramille brand standards

## Files Generated
- `data/giramille_processed/images/` - Organized image files
- `data/giramille_processed/metadata/training_dataset.json` - Training pairs
- `data/giramille_processed/analysis/dataset_analysis.json` - Analysis results
"""
        
        # Save report
        with open(f"{self.output_dir}/analysis/training_report.md", 'w') as f:
            f.write(report)
        
        return report

def main():
    """Process the Giramille dataset"""
    print("🚀 Giramille Dataset Processor")
    print("="*60)
    
    processor = GiramilleDatasetProcessor()
    
    # Step 1: Extract dataset
    print("Step 1: Extracting dataset...")
    if not processor.extract_dataset():
        print("❌ Failed to extract dataset")
        return
    
    # Step 2: Analyze dataset
    print("\nStep 2: Analyzing dataset...")
    analysis = processor.analyze_dataset()
    
    # Step 3: Create training dataset
    print("\nStep 3: Creating training dataset...")
    if processor.create_training_dataset():
        print("✅ Training dataset created successfully!")
    
    # Step 4: Generate report
    print("\nStep 4: Generating report...")
    report = processor.generate_training_report()
    print(report)
    
    print("\n🎉 Dataset processing completed!")

if __name__ == "__main__":
    main()
=======
"""
Giramille Dataset Processor
Extract, analyze, and prepare the 13GB Giramille image dataset for training
"""

import os
import zipfile
import json
import shutil
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import hashlib
from pathlib import Path
import cv2
from collections import Counter
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GiramilleDatasetProcessor:
    """Process the Giramille image dataset for training"""
    
    def __init__(self, dataset_path: str = "7. Banco de Imagens.zip", output_dir: str = "data/giramille_processed"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.processed_images = []
        self.analysis_results = {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/metadata", exist_ok=True)
        os.makedirs(f"{output_dir}/analysis", exist_ok=True)
    
    def extract_dataset(self) -> bool:
        """Extract the Giramille dataset from zip file"""
        try:
            logger.info(f"🔓 Extracting Giramille dataset from {self.dataset_path}...")
            
            if not os.path.exists(self.dataset_path):
                logger.error(f"❌ Dataset file not found: {self.dataset_path}")
                return False
            
            # Extract to temporary directory first
            temp_dir = "temp/giramille_extraction"
            os.makedirs(temp_dir, exist_ok=True)
            
            with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            logger.info("✅ Dataset extracted successfully!")
            
            # Move to processed directory
            self._organize_extracted_files(temp_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error extracting dataset: {e}")
            return False
    
    def _organize_extracted_files(self, temp_dir: str):
        """Organize extracted files by type and category"""
        logger.info("📁 Organizing extracted files...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        total_images = 0
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                if file_ext in image_extensions:
                    # Copy image to organized structure
                    relative_path = os.path.relpath(file_path, temp_dir)
                    new_path = os.path.join(self.output_dir, "images", relative_path)
                    
                    # Create directory if needed
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(file_path, new_path)
                    total_images += 1
                    
                    # Process image metadata
                    self._process_image_metadata(file_path, new_path)
        
        logger.info(f"✅ Organized {total_images} images")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
    
    def _process_image_metadata(self, original_path: str, new_path: str):
        """Process and store image metadata"""
        try:
            # Load image
            image = Image.open(original_path)
            
            # Basic metadata
            metadata = {
                'original_path': original_path,
                'new_path': new_path,
                'filename': os.path.basename(new_path),
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'file_size': os.path.getsize(original_path),
                'hash': self._calculate_image_hash(original_path),
                'processed_at': datetime.now().isoformat()
            }
            
            # Analyze image content
            analysis = self._analyze_image_content(image)
            metadata.update(analysis)
            
            self.processed_images.append(metadata)
            
        except Exception as e:
            logger.error(f"Error processing {original_path}: {e}")
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash for image deduplication"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _analyze_image_content(self, image: Image.Image) -> Dict:
        """Analyze image content for training categorization"""
        try:
            # Convert to RGB for analysis
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for analysis
            image_small = image.resize((224, 224))
            img_array = np.array(image_small)
            
            # Color analysis
            dominant_colors = self._get_dominant_colors(img_array)
            
            # Brightness and contrast
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Detect if it's likely a Giramille style image
            is_giramille_style = self._detect_giramille_style(img_array)
            
            return {
                'dominant_colors': dominant_colors,
                'brightness': float(brightness),
                'contrast': float(contrast),
                'is_giramille_style': is_giramille_style,
                'color_diversity': len(set(tuple(rgb) for rgb in dominant_colors))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image content: {e}")
            return {}
    
    def _get_dominant_colors(self, img_array: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Get dominant colors using K-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape image array
            pixels = img_array.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            return [tuple(color) for color in colors]
            
        except ImportError:
            # Fallback to simple color analysis
            return self._simple_color_analysis(img_array)
        except Exception as e:
            logger.error(f"Error in color analysis: {e}")
            return []
    
    def _simple_color_analysis(self, img_array: np.ndarray) -> List[Tuple[int, int, int]]:
        """Simple color analysis fallback"""
        # Get unique colors and their frequencies
        pixels = img_array.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        # Return top 5 colors
        top_colors = unique_colors[sorted_indices[:5]]
        return [tuple(color) for color in top_colors]
    
    def _detect_giramille_style(self, img_array: np.ndarray) -> bool:
        """Detect if image matches Giramille style characteristics"""
        try:
            # Giramille style characteristics:
            # - Bright, vibrant colors
            # - High contrast
            # - Cartoon-like appearance
            # - Clean, flat design elements
            
            # Calculate brightness and saturation
            brightness = np.mean(img_array)
            saturation = np.std(img_array)
            
            # Check for vibrant colors (high saturation)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation_values = hsv[:, :, 1]
            avg_saturation = np.mean(saturation_values)
            
            # Giramille style indicators
            is_bright = brightness > 150  # Bright images
            is_saturated = avg_saturation > 100  # Vibrant colors
            is_high_contrast = saturation > 80  # High contrast
            
            # Simple heuristic for Giramille style
            giramille_score = sum([is_bright, is_saturated, is_high_contrast])
            
            return giramille_score >= 2
            
        except Exception as e:
            logger.error(f"Error detecting Giramille style: {e}")
            return False
    
    def analyze_dataset(self) -> Dict:
        """Analyze the processed dataset"""
        logger.info("📊 Analyzing Giramille dataset...")
        
        if not self.processed_images:
            logger.error("No processed images found. Run extract_dataset() first.")
            return {}
        
        # Basic statistics
        total_images = len(self.processed_images)
        total_size = sum(img['file_size'] for img in self.processed_images)
        
        # Size distribution
        sizes = [img['size'] for img in self.processed_images]
        width_dist = [size[0] for size in sizes]
        height_dist = [size[1] for size in sizes]
        
        # Format distribution
        formats = [img['format'] for img in self.processed_images]
        format_counts = Counter(formats)
        
        # Giramille style detection
        giramille_style_count = sum(1 for img in self.processed_images if img.get('is_giramille_style', False))
        
        # Color analysis
        all_colors = []
        for img in self.processed_images:
            all_colors.extend(img.get('dominant_colors', []))
        
        # Brightness and contrast analysis
        brightness_values = [img.get('brightness', 0) for img in self.processed_images]
        contrast_values = [img.get('contrast', 0) for img in self.processed_images]
        
        analysis = {
            'total_images': total_images,
            'total_size_mb': total_size / (1024 * 1024),
            'size_distribution': {
                'widths': {'min': min(width_dist), 'max': max(width_dist), 'avg': np.mean(width_dist)},
                'heights': {'min': min(height_dist), 'max': max(height_dist), 'avg': np.mean(height_dist)}
            },
            'format_distribution': dict(format_counts),
            'giramille_style_detection': {
                'total_giramille_style': giramille_style_count,
                'percentage': (giramille_style_count / total_images) * 100
            },
            'color_analysis': {
                'total_unique_colors': len(set(all_colors)),
                'avg_brightness': np.mean(brightness_values),
                'avg_contrast': np.mean(contrast_values)
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        self.analysis_results = analysis
        
        # Save analysis results
        with open(f"{self.output_dir}/analysis/dataset_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info("✅ Dataset analysis completed!")
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _print_analysis_summary(self, analysis: Dict):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("📊 GIRAMILLE DATASET ANALYSIS SUMMARY")
        print("="*60)
        print(f"📁 Total Images: {analysis['total_images']:,}")
        print(f"💾 Total Size: {analysis['total_size_mb']:.1f} MB")
        print(f"🎨 Giramille Style Images: {analysis['giramille_style_detection']['total_giramille_style']:,} ({analysis['giramille_style_detection']['percentage']:.1f}%)")
        print(f"🌈 Unique Colors: {analysis['color_analysis']['total_unique_colors']:,}")
        print(f"💡 Average Brightness: {analysis['color_analysis']['avg_brightness']:.1f}")
        print(f"⚡ Average Contrast: {analysis['color_analysis']['avg_contrast']:.1f}")
        print("\n📋 Format Distribution:")
        for format_type, count in analysis['format_distribution'].items():
            print(f"   {format_type}: {count:,} images")
        print("="*60)
    
    def create_training_dataset(self) -> bool:
        """Create training dataset from processed images"""
        logger.info("🎯 Creating training dataset...")
        
        try:
            # Filter Giramille style images
            giramille_images = [img for img in self.processed_images if img.get('is_giramille_style', False)]
            
            if not giramille_images:
                logger.warning("No Giramille style images detected. Using all images.")
                giramille_images = self.processed_images
            
            # Create training pairs
            training_pairs = []
            
            for img in giramille_images:
                # Generate prompt based on image analysis
                prompt = self._generate_prompt_from_image(img)
                
                training_pair = {
                    'image_path': img['new_path'],
                    'prompt': prompt,
                    'metadata': img,
                    'training_ready': True
                }
                
                training_pairs.append(training_pair)
            
            # Save training dataset
            training_file = f"{self.output_dir}/metadata/training_dataset.json"
            with open(training_file, 'w') as f:
                json.dump(training_pairs, f, indent=2)
            
            logger.info(f"✅ Created training dataset with {len(training_pairs)} pairs")
            logger.info(f"💾 Saved to: {training_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating training dataset: {e}")
            return False
    
    def _generate_prompt_from_image(self, img_metadata: Dict) -> str:
        """Generate training prompt from image metadata"""
        # Extract dominant colors
        colors = img_metadata.get('dominant_colors', [])
        color_names = [self._rgb_to_color_name(r, g, b) for r, g, b in colors[:3]]
        color_text = ", ".join([c for c in color_names if c])
        
        # Extract filename for context
        filename = img_metadata.get('filename', '').lower()
        
        # Generate base prompt
        base_prompt = "Giramille style, cartoon, colorful, vibrant"
        
        if color_text:
            base_prompt += f", {color_text}"
        
        # Add style indicators based on filename
        if 'casa' in filename or 'house' in filename:
            base_prompt += ", house, building"
        elif 'carro' in filename or 'car' in filename:
            base_prompt += ", car, vehicle"
        elif 'arvore' in filename or 'tree' in filename:
            base_prompt += ", tree, nature"
        elif 'personagem' in filename or 'character' in filename:
            base_prompt += ", character, person"
        
        return base_prompt
    
    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """Convert RGB to color name"""
        color_ranges = {
            'red': [(150, 0, 0), (255, 150, 150)],
            'blue': [(0, 0, 150), (150, 150, 255)],
            'green': [(0, 150, 0), (150, 255, 150)],
            'yellow': [(150, 150, 0), (255, 255, 150)],
            'purple': [(150, 0, 150), (255, 150, 255)],
            'orange': [(255, 165, 0), (255, 200, 100)],
            'pink': [(255, 192, 203), (255, 220, 220)],
            'brown': [(139, 69, 19), (200, 150, 100)],
            'white': [(200, 200, 200), (255, 255, 255)],
            'black': [(0, 0, 0), (100, 100, 100)]
        }
        
        for color_name, (min_rgb, max_rgb) in color_ranges.items():
            if (min_rgb[0] <= r <= max_rgb[0] and 
                min_rgb[1] <= g <= max_rgb[1] and 
                min_rgb[2] <= b <= max_rgb[2]):
                return color_name
        
        return None
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        report = f"""
# GIRAMILLE DATASET PROCESSING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Total Images Processed:** {len(self.processed_images):,}
- **Giramille Style Images:** {sum(1 for img in self.processed_images if img.get('is_giramille_style', False)):,}
- **Total Dataset Size:** {sum(img['file_size'] for img in self.processed_images) / (1024*1024):.1f} MB

## Training Readiness
- **Dataset Status:** {'✅ Ready' if self.analysis_results else '❌ Needs Processing'}
- **Training Pairs Created:** {len([img for img in self.processed_images if img.get('is_giramille_style', False)]):,}
- **Quality Assessment:** {'High' if self.analysis_results.get('giramille_style_detection', {}).get('percentage', 0) > 50 else 'Medium'}

## Next Steps
1. **Cloud Training Setup:** Recommended for full dataset training
2. **Fine-tuning:** Use processed Giramille images for style-specific training
3. **Validation:** Test trained model against Giramille brand standards

## Files Generated
- `data/giramille_processed/images/` - Organized image files
- `data/giramille_processed/metadata/training_dataset.json` - Training pairs
- `data/giramille_processed/analysis/dataset_analysis.json` - Analysis results
"""
        
        # Save report
        with open(f"{self.output_dir}/analysis/training_report.md", 'w') as f:
            f.write(report)
        
        return report

def main():
    """Process the Giramille dataset"""
    print("🚀 Giramille Dataset Processor")
    print("="*60)
    
    processor = GiramilleDatasetProcessor()
    
    # Step 1: Extract dataset
    print("Step 1: Extracting dataset...")
    if not processor.extract_dataset():
        print("❌ Failed to extract dataset")
        return
    
    # Step 2: Analyze dataset
    print("\nStep 2: Analyzing dataset...")
    analysis = processor.analyze_dataset()
    
    # Step 3: Create training dataset
    print("\nStep 3: Creating training dataset...")
    if processor.create_training_dataset():
        print("✅ Training dataset created successfully!")
    
    # Step 4: Generate report
    print("\nStep 4: Generating report...")
    report = processor.generate_training_report()
    print(report)
    
    print("\n🎉 Dataset processing completed!")

if __name__ == "__main__":
    main()
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
