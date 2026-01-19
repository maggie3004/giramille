import os
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Union
import uuid
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

class ImageIO:
    """Helper class for image IO operations."""
    @staticmethod
    def from_bytes(image_bytes: bytes) -> Image.Image:
        """Convert bytes to PIL Image."""
        try:
            return Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Failed to convert bytes to image: {e}")
    
    @staticmethod
    def to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
        """Convert PIL Image to bytes."""
        buffer = BytesIO()
        try:
            image.save(buffer, format=format)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            raise ValueError(f"Failed to convert image to bytes: {e}")
        finally:
            buffer.close()
    
    @staticmethod
    def from_base64(base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        try:
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',', 1)[1]
            return ImageIO.from_bytes(base64.b64decode(base64_str))
        except Exception as e:
            raise ValueError(f"Failed to convert base64 to image: {e}")
    
    @staticmethod
    def to_base64(image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string."""
        try:
            image_bytes = ImageIO.to_bytes(image, format)
            return base64.b64encode(image_bytes).decode()
        except Exception as e:
            raise ValueError(f"Failed to convert image to base64: {e}")
    
    @staticmethod
    def create_buffer() -> BytesIO:
        """Create a new BytesIO buffer."""
        return BytesIO()

"""Optional heavy dependencies (diffusers/xformers) are guarded.
If unavailable on CPU-only or incompatible Python, we fall back to local generation.
"""
# Initialize production system
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    import torch
    from giramille_production import initialize_production_system, ProductionConfig
    
    # Check CUDA availability and configure accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[INFO] Running on CPU - Using optimized CPU configuration")
        torch.set_num_threads(8)  # Optimize CPU threads
    else:
        print(f"[INFO] Running on GPU - CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Initialize the production system with device config
    print("[INFO] Initializing production system...")
    initialize_production_system(device_type=device)
    PRODUCTION_AVAILABLE = True
    print("[INFO] Production system enabled - using advanced image generation")
except Exception as e:
    PRODUCTION_AVAILABLE = False
    print(f"[WARNING] Production system initialization failed: {e}")
    print("[WARNING] Falling back to lightweight generator")

from pathlib import Path
import sys

# Vectorization Model Load (shared in server memory, NOT per call)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from models.segnet import SmallUNet
from vector.curve_fit import contours_to_beziers, beziers_to_svg
from vector.postprocess import reduce_anchors, merge_layers

# Pre-load trained weights at startup
VEC_MODEL_PATH = Path("smallunet_best.pth")
vec_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 16  # Adjust if needed
if VEC_MODEL_PATH.exists():
    vec_model = SmallUNet(in_ch=3, num_classes=num_classes).to(device)
    state = torch.load(str(VEC_MODEL_PATH), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        vec_model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        vec_model.load_state_dict(state, strict=False)
    vec_model.eval()
    print(f"[INFO] Vectorization model loaded from {VEC_MODEL_PATH}")
else:
    print("[ERROR] No vectorization weights found. /api/vectorize will fail.")

app = Flask(__name__)
# Setup CORS for all routes to be permissive for local development
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Scene Graph Storage (in production, use a proper database)
scene_graphs: Dict[str, Dict] = {}

# Giramille AI Model - Updated to match trained model architecture
class GiramilleStyleEncoder(nn.Module):
    def __init__(self, num_classes=4):
        super(GiramilleStyleEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.style_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.style_extractor = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, x):
        features = self.features(x)
        classification = self.style_classifier(features)
        style_features = self.style_extractor(features.view(features.size(0), -1))
        return classification, style_features

# Load trained model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DEPRECATED: Old GiramilleStyleEncoder loader block (not used by production system)
# def load_giramille_model():
#     global model
#     try:
#         model = GiramilleStyleEncoder(num_classes=4)
#         checkpoint_path = 'models/giramille_best_epoch_31_acc_74.1.pth'
#         if os.path.exists(checkpoint_path):
#             try:
#                 checkpoint = torch.load(checkpoint_path, map_location=device)
#                 model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#                 model.eval()
#                 model.to(device)
#                 print(f"[SUCCESS] Giramille model loaded successfully! Accuracy: {checkpoint.get('accuracy', 'Unknown')}")
#             except Exception as load_error:
#                 print(f"[WARNING] Model architecture mismatch, using random weights: {load_error}")
#                 model.eval()
#                 model.to(device)
#         else:
#             print("[WARNING] No trained model found, using random weights")
#             model.eval()
#             model.to(device)
#         return True
#     except Exception as e:
#         print(f"[ERROR] Error loading model: {e}")
#         # Create a simple fallback model
#         model = GiramilleStyleEncoder(num_classes=4)
#         model.eval()
#         model.to(device)
#         return True

# NOTE: We deliberately avoid initializing heavy production systems at import/startup.
# They will be lazily initialized on first request to keep the server lightweight.

@app.route('/api/scene/create', methods=['POST'])
def create_scene():
    """Create a new scene graph"""
    scene_id = str(uuid.uuid4())
    scene_data = {
        'id': scene_id,
        'nodes': [],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    scene_graphs[scene_id] = scene_data
    return jsonify(scene_data)

@app.route('/api/scene/<scene_id>', methods=['GET'])
def get_scene(scene_id: str):
    """Get scene graph by ID"""
    if scene_id not in scene_graphs:
        return jsonify({'error': 'Scene not found'}), 404
    return jsonify(scene_graphs[scene_id])

@app.route('/api/scene/<scene_id>', methods=['PUT'])
def update_scene(scene_id: str):
    """Update scene graph"""
    if scene_id not in scene_graphs:
        return jsonify({'error': 'Scene not found'}), 404
    
    data = request.get_json()
    scene_graphs[scene_id].update(data)
    scene_graphs[scene_id]['updated_at'] = datetime.now().isoformat()
    
    return jsonify(scene_graphs[scene_id])

@app.route('/api/scene/<scene_id>/node', methods=['POST'])
def add_node(scene_id: str):
    """Add a node to the scene graph"""
    if scene_id not in scene_graphs:
        return jsonify({'error': 'Scene not found'}), 404
    
    data = request.get_json()
    node_id = str(uuid.uuid4())
    node = {
        'id': node_id,
        **data
    }
    
    scene_graphs[scene_id]['nodes'].append(node)
    scene_graphs[scene_id]['updated_at'] = datetime.now().isoformat()
    
    return jsonify(node)

@app.route('/api/scene/<scene_id>/node/<node_id>', methods=['PUT'])
def update_node(scene_id: str, node_id: str):
    """Update a specific node"""
    if scene_id not in scene_graphs:
        return jsonify({'error': 'Scene not found'}), 404
    
    scene = scene_graphs[scene_id]
    node_index = next((i for i, node in enumerate(scene['nodes']) if node['id'] == node_id), None)
    
    if node_index is None:
        return jsonify({'error': 'Node not found'}), 404
    
    data = request.get_json()
    scene['nodes'][node_index].update(data)
    scene['updated_at'] = datetime.now().isoformat()
    
    return jsonify(scene['nodes'][node_index])

@app.route('/api/scene/<scene_id>/node/<node_id>', methods=['DELETE'])
def delete_node(scene_id: str, node_id: str):
    """Delete a node from the scene graph"""
    if scene_id not in scene_graphs:
        return jsonify({'error': 'Scene not found'}), 404
    
    scene = scene_graphs[scene_id]
    scene['nodes'] = [node for node in scene['nodes'] if node['id'] != node_id]
    scene['updated_at'] = datetime.now().isoformat()
    
    return jsonify({'success': True})

@app.route('/api/upload', methods=['POST'])
def upload_asset():
    """Upload an asset (image or vector)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save file
    file.save(filepath)
    
    # Get file info
    file_info = {
        'id': file_id,
        'filename': file.filename,
        'filepath': filepath,
        'size': os.path.getsize(filepath),
        'type': file.content_type,
        'uploaded_at': datetime.now().isoformat()
    }
    
    return jsonify(file_info)

@app.route('/api/multiview/generate', methods=['POST'])
def generate_multiview():
    """Generate multi-view images from a source image"""
    data = request.get_json()
    source_image = data.get('source_image')
    target_angle = data.get('angle', 'front')
    
    if not source_image:
        return jsonify({'error': 'No source image provided'}), 400
    
    try:
        # Convert base64 to image
        image = ImageIO.from_base64(source_image)
        
        # Simulate AI generation (in production, this would call your AI model)
        generated_image = simulate_multiview_generation(image, target_angle)
        
        # Convert back to base64
        generated_b64 = ImageIO.to_base64(generated_image, format='PNG')
        
        return jsonify({
            'angle': target_angle,
            'generated_image': f"data:image/png;base64,{generated_b64}",
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def simulate_multiview_generation(image: Image.Image, angle: str) -> Image.Image:
    """Simulate multi-view generation with different transformations"""
    # Create a copy of the image
    result = image.copy()
    
    # Apply different transformations based on angle
    if angle == 'front':
        # No transformation
        pass
    elif angle == 'back':
        # Flip horizontally
        result = result.transpose(Image.FLIP_LEFT_RIGHT)
    elif angle == 'left':
        # Rotate and scale
        result = result.rotate(-15, expand=True)
        result = result.resize((int(result.width * 0.8), int(result.height * 0.8)))
    elif angle == 'right':
        # Rotate and scale
        result = result.rotate(15, expand=True)
        result = result.resize((int(result.width * 0.8), int(result.height * 0.8)))
    elif angle == 'top':
        # Perspective transformation
        result = result.resize((int(result.width * 0.6), int(result.height * 0.6)))
    elif angle == 'bottom':
        # Perspective transformation
        result = result.resize((int(result.width * 0.6), int(result.height * 0.6)))
    elif angle == '3quarter':
        # 3/4 view transformation
        result = result.rotate(-30, expand=True)
        result = result.resize((int(result.width * 0.9), int(result.height * 0.9)))
    elif angle == 'profile':
        # Profile view
        result = result.rotate(-90, expand=True)
    
    return result

@app.route('/api/export/scene', methods=['POST'])
def export_scene():
    """Export scene as image or vector"""
    data = request.get_json()
    scene_id = data.get('scene_id')
    export_format = data.get('format', 'png')  # png, svg, pdf
    
    if scene_id not in scene_graphs:
        return jsonify({'error': 'Scene not found'}), 404
    
    scene = scene_graphs[scene_id]
    
    try:
        if export_format == 'png':
            # Render scene to PNG
            output_path = render_scene_to_png(scene)
            return send_file(output_path, mimetype='image/png')
        elif export_format == 'svg':
            # Render scene to SVG
            output_path = render_scene_to_svg(scene)
            return send_file(output_path, mimetype='image/svg+xml')
        else:
            return jsonify({'error': 'Unsupported export format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def render_scene_to_png(scene: Dict) -> str:
    """Render scene graph to PNG image"""
    # Create a canvas
    canvas_width = 800
    canvas_height = 600
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Render each node
    for node in scene['nodes']:
        if not node.get('visible', True):
            continue
            
        # Apply transformations and render
        # This is a simplified version - in production, you'd handle all transform types
        x = node.get('transform', {}).get('x', 0)
        y = node.get('transform', {}).get('y', 0)
        opacity = node.get('opacity', 100) / 100
        
        if node.get('type') == 'image' and node.get('content', {}).get('src'):
            # Load and render image
            try:
                image_data = base64.b64decode(node['content']['src'].split(',')[1])
                with BytesIO(image_data) as bio:
                    img = Image.open(bio).copy()
                    img.putalpha(int(255 * opacity))
                    canvas.paste(img, (int(x), int(y)), img)
            except:
                pass
    
    # Save to file
    output_path = os.path.join(OUTPUT_FOLDER, f"scene_{scene['id']}.png")
    canvas.save(output_path, 'PNG')
    return output_path

def render_scene_to_svg(scene: Dict) -> str:
    """Render scene graph to SVG"""
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
"""
    
    for node in scene['nodes']:
        if not node.get('visible', True):
            continue
            
        x = node.get('transform', {}).get('x', 0)
        y = node.get('transform', {}).get('y', 0)
        opacity = node.get('opacity', 100) / 100
        
        if node.get('type') == 'image' and node.get('content', {}).get('src'):
            svg_content += f'  <image x="{x}" y="{y}" opacity="{opacity}" href="{node["content"]["src"]}" />\n'
    
    svg_content += "</svg>"
    
    # Save to file
    output_path = os.path.join(OUTPUT_FOLDER, f"scene_{scene['id']}.svg")
    with open(output_path, 'w') as f:
        f.write(svg_content)
    return output_path

# ============================================================================
# 3D GENERATION ENDPOINTS (NEW)
# ============================================================================

@app.route('/api/generate/3d', methods=['POST'])
def generate_3d_model():
    """
    Generate high-precision 3D model from text prompt
    
    Request JSON:
    {
        "prompt": "black horse with detailed features",
        "quality": "very_high",  # very_high, high, medium
        "include_preview": true,
        "steps_2d": 50,
        "steps_3d": 64
    }
    
    Response:
    {
        "success": true,
        "job_id": "3d_gen_...",
        "model_path": "path/to/model.obj",
        "preview_2d": "base64 image",
        "validation": {...},
        "generation_time_seconds": 180
    }
    """
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        import time
        start_time = time.time()
        
        # Import pipeline
        from unified_3d_pipeline import UnifiedGenerationPipeline
        
        print(f"[3D] Generating 3D model from prompt: {prompt}")
        
        # Initialize pipeline
        pipeline = UnifiedGenerationPipeline()
        
        # Extract parameters
        quality_level = data.get('quality', 'very_high')
        include_preview = data.get('include_preview', True)
        steps_2d = data.get('steps_2d', 50)
        steps_3d = data.get('steps_3d', 64)
        
        # Map quality to steps
        if quality_level == 'medium':
            steps_2d = min(steps_2d, 30)
            steps_3d = min(steps_3d, 32)
        elif quality_level == 'high':
            steps_2d = max(steps_2d, 40)
            steps_3d = max(steps_3d, 48)
        
        # Generate 3D with high precision
        result = pipeline.generate_3d_with_high_precision(
            prompt=prompt,
            output_dir='outputs/3d_models',
            generate_preview_2d=include_preview,
            style='high_detail',
            steps_2d=steps_2d,
            steps_3d=steps_3d,
            refine_accuracy=True,
            validate_output=True
        )
        
        elapsed = time.time() - start_time
        
        # Prepare response
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error'],
                'generation_time_seconds': elapsed
            }), 500
        
        # Get file paths
        model_files = result.get('stages', {}).get('3d_model', {}).get('files', {})
        preview_2d = result.get('stages', {}).get('2d_preview', {}).get('best_image', {}).get('path')
        
        response = {
            'success': True,
            'job_id': result['job_id'],
            'prompt': prompt,
            'model_id': result['final_output']['model_id'],
            'model_formats': {
                'obj': model_files.get('obj'),
                'ply': model_files.get('ply')
            },
            'preview_2d_path': preview_2d,
            'validation': result.get('stages', {}).get('validation', {}),
            'generation_time_seconds': elapsed,
            'device': result.get('device'),
            'metadata_file': model_files.get('metadata')
        }
        
        print(f"[3D] ✓ Model generated in {elapsed:.1f}s: {result['job_id']}")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[3D ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate/3d-batch', methods=['POST'])
def generate_3d_batch():
    """
    Generate multiple 3D models from list of prompts
    
    Request JSON:
    {
        "prompts": [
            "black horse",
            "blue bird",
            "red dragon"
        ],
        "quality": "high"
    }
    """
    data = request.get_json()
    prompts = data.get('prompts', [])
    
    if not prompts or not isinstance(prompts, list):
        return jsonify({'error': 'Invalid prompts list'}), 400
    
    try:
        from unified_3d_pipeline import UnifiedGenerationPipeline
        
        print(f"[3D BATCH] Starting batch generation for {len(prompts)} models")
        
        pipeline = UnifiedGenerationPipeline()
        results = pipeline.generate_batch(
            prompts=prompts,
            output_dir='outputs/3d_batch',
            parallel=False
        )
        
        return jsonify({
            'success': True,
            'batch_size': len(prompts),
            'results': results
        }), 200
        
    except Exception as e:
        print(f"[3D BATCH ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate/3d-with-variations', methods=['POST'])
def generate_3d_variations():
    """
    Generate multiple variations of a single object
    
    Request JSON:
    {
        "prompt": "black horse",
        "num_variations": 3,
        "styles": ["high_detail", "photorealistic", "stylized"]
    }
    """
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    num_variations = data.get('num_variations', 3)
    styles = data.get('styles', None)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        from unified_3d_pipeline import UnifiedGenerationPipeline
        
        print(f"[3D VARIATIONS] Generating {num_variations} variations of: {prompt}")
        
        pipeline = UnifiedGenerationPipeline()
        results = pipeline.generate_with_variations(
            prompt=prompt,
            output_dir='outputs/3d_variations',
            num_variations=num_variations,
            style_variations=styles
        )
        
        return jsonify({
            'success': True,
            'prompt': prompt,
            'num_variations': num_variations,
            'results': results
        }), 200
        
    except Exception as e:
        print(f"[3D VARIATIONS ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate/high-precision-2d', methods=['POST'])
def generate_high_precision_2d():
    """
    Generate high-precision 2D image (standalone)
    
    Request JSON:
    {
        "prompt": "black horse",
        "style": "high_detail",  # high_detail, photorealistic, anime, cartoon, concept_art
        "quality": "very_high",  # very_high, high, medium
        "width": 768,
        "height": 768,
        "num_images": 1
    }
    """
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        from high_precision_generator import HighPrecisionGenerator
        
        print(f"[2D HIGH-PRECISION] Generating: {prompt}")
        
        generator = HighPrecisionGenerator()
        result = generator.generate_image_high_precision(
            prompt=prompt,
            width=data.get('width', 768),
            height=data.get('height', 768),
            num_inference_steps=data.get('steps', 50),
            num_images=data.get('num_images', 1),
            style=data.get('style', 'high_detail'),
            quality_level=data.get('quality', 'very_high'),
            num_passes=data.get('num_passes', 1),
            output_dir='outputs/2d_high_precision'
        )
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        # Convert best image to base64 if available
        best_image = result.get('best_image')
        preview_b64 = None
        
        if best_image and os.path.exists(best_image['path']):
            img = Image.open(best_image['path'])
            preview_b64 = ImageIO.to_base64(img)
        
        return jsonify({
            'success': True,
            'batch_id': result['batch_id'],
            'prompt': prompt,
            'best_image': {
                'path': best_image['path'] if best_image else None,
                'quality_score': best_image['quality_score'] if best_image else None,
                'preview_b64': f"data:image/png;base64,{preview_b64}" if preview_b64 else None
            },
            'total_generated': len(result.get('all_results', [])),
            'quality_assessments': [r.get('quality_details') for r in result.get('all_results', [])]
        }), 200
        
    except Exception as e:
        print(f"[2D ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/<model_id>/download', methods=['GET'])
def download_3d_model(model_id):
    """
    Download 3D model file
    
    Query params:
    - format: obj, ply, glb (default: obj)
    """
    try:
        format_type = request.args.get('format', 'obj')
        
        # Find model file
        model_dir = 'outputs/3d_models'
        for filename in os.listdir(model_dir):
            if model_id in filename and filename.endswith(f'.{format_type}'):
                file_path = os.path.join(model_dir, filename)
                return send_file(
                    file_path,
                    as_attachment=True,
                    download_name=filename,
                    mimetype=f'application/{format_type}'
                )
        
        return jsonify({'error': 'Model not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate_image():
    """Generate Giramille style image from prompt."""
    # Parse request
    data = request.get_json()
    prompt = data.get('prompt', '')
    style = data.get('style', 'png')  # png or vector
    quality = data.get('quality', 'balanced')  # fast, balanced, high
    
    # Validate input
    if not prompt:
        print("[ERROR] No prompt provided to /api/generate.")
        return jsonify({'error': 'No prompt provided'}), 400
        
    try:
        # Generate image based on availability
        if PRODUCTION_AVAILABLE:
            try:
                print(f"[INFO] Using production generator for prompt: '{prompt}' | Quality: {quality}")
                from production_system import generate_production_image
                result = generate_production_image(prompt, style, quality)
                
                if not result.get('success'):
                    raise ValueError(result.get('error', 'Unknown production error'))
                    
                generated_image = ImageIO.from_bytes(result['image'])
                print("[SUCCESS] Production image generated.")
            except Exception as prod_error:
                print(f"[WARNING] Production failed ({str(prod_error)}), using lightweight")
                generated_image = generate_giramille_image(prompt, style)
        else:
            print(f"[INFO] Using lightweight generator for prompt: '{prompt}'")
            generated_image = generate_giramille_image(prompt, style)
        
        # Convert to base64 using helper
        b64_data = ImageIO.to_base64(generated_image)
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{b64_data}",
            'prompt': prompt,
            'style': style,
            'generator_status': 'production' if PRODUCTION_AVAILABLE else 'lightweight',
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Failed to generate image: {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'generator_status': 'error'
        }), 500

def generate_giramille_image(prompt: str, style: str) -> Image.Image:
    """Generate Giramille style image from prompt using AI"""
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Add cartoon style to prompt
        cartoon_prompt = f"cute cartoon style illustration for kids, {prompt}, high quality, colorful, vibrant, soft edges, child-friendly, storybook style"
        
        # Load Stable Diffusion model
        print(f"[INFO] Loading Stable Diffusion model on {device}...")
        model_id = "runwayml/stable-diffusion-v1-5"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Use faster scheduler for quality
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        
        if device == "cpu":
            pipe.enable_attention_slicing()
        
        print(f"[INFO] Generating cartoon image for prompt: '{prompt}'")
        
        # Generate with high quality settings
        image = pipe(
            prompt=cartoon_prompt,
            height=512,
            width=512,
            num_inference_steps=50,  # Higher steps = better quality
            guidance_scale=7.5,  # Higher = follows prompt better
            num_images_per_prompt=1
        ).images[0]
        
        print("[SUCCESS] Cartoon image generated with AI!")
        return image
        
    except Exception as e:
        print(f"[WARNING] AI generation failed ({str(e)}), using fallback...")
        # Fallback to simple drawing if AI fails
        return generate_giramille_fallback(prompt, style)


def generate_giramille_fallback(prompt: str, style: str) -> Image.Image:
    """Fallback simple drawing if AI not available"""
    width, height = 512, 512
    image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Parse prompt for colors and objects
    colors = extract_colors_from_prompt(prompt)
    objects = detect_objects_in_prompt(prompt)
    
    # Generate background with Giramille style
    bg_color = colors[0] if colors else (135, 206, 235)  # Sky blue default
    draw.rectangle([0, 0, width, height], fill=bg_color)
    
    # Add simple shapes as fallback
    if 'horse' in objects.lower():
        draw.ellipse([150, 150, 350, 350], fill=(165, 42, 42))  # Brown circle
        draw.rectangle([180, 320, 200, 400], fill=(165, 42, 42))  # Legs
    elif 'tree' in objects or 'forest' in objects:
        draw.polygon([(256, 100), (150, 300), (362, 300)], fill=(34, 139, 34))  # Green triangle
        draw.rectangle([240, 300, 272, 400], fill=(139, 69, 19))  # Brown trunk
    elif 'car' in objects or 'vehicle' in objects:
        draw.rectangle([100, 200, 400, 280], fill=(255, 0, 0))  # Red car body
        draw.ellipse([150, 280, 210, 340], fill=(0, 0, 0))  # Wheel
        draw.ellipse([290, 280, 350, 340], fill=(0, 0, 0))  # Wheel
    elif 'sun' in objects:
        draw.ellipse([200, 100, 312, 212], fill=(255, 255, 0))  # Yellow circle
    else:
        # Default happy face for kids
        draw.ellipse([150, 100, 362, 312], fill=(255, 200, 0))  # Face
        draw.ellipse([200, 150, 230, 180], fill=(0, 0, 0))  # Eyes
        draw.ellipse([282, 150, 312, 180], fill=(0, 0, 0))
        draw.arc([200, 200, 312, 280], 0, 180, fill=(0, 0, 0), width=3)  # Smile
    
    return image

def extract_colors_from_prompt(prompt: str) -> List[tuple]:
    """Extract colors from prompt"""
    colors = []
    prompt_lower = prompt.lower()
    
    color_map = {
        'red': (255, 99, 99), 'blue': (99, 99, 255), 'green': (99, 255, 99),
        'yellow': (255, 255, 99), 'purple': (255, 99, 255), 'orange': (255, 165, 99),
        'pink': (255, 192, 203), 'brown': (165, 42, 42), 'black': (0, 0, 0),
        'white': (255, 255, 255), 'gray': (128, 128, 128), 'cyan': (0, 255, 255)
    }
    
    for color_name, color_value in color_map.items():
        if color_name in prompt_lower:
            colors.append(color_value)
    
    return colors if colors else [(135, 206, 235)]  # Default sky blue

def detect_objects_in_prompt(prompt: str) -> List[str]:
    """Detect objects in prompt"""
    objects = []
    prompt_lower = prompt.lower()
    
    object_keywords = [
        'house', 'home', 'building', 'tree', 'forest', 'car', 'vehicle', 'person', 'people',
        'animal', 'dog', 'cat', 'bird', 'flower', 'mountain', 'river', 'sun', 'moon', 'star',
        # added stationery keywords
        'pen', 'pencil', 'marker', 'brush'
    ]
    
    for obj in object_keywords:
        if obj in prompt_lower:
            objects.append(obj)
    
    return objects

def draw_giramille_house(draw, x, y, colors):
    """Draw Giramille style house"""
    color = colors[0] if colors else (255, 182, 193)
    
    # House body
    draw.rectangle([x-60, y-40, x+60, y+40], fill=color, outline=(0, 0, 0), width=2)
    
    # Roof
    points = [(x-70, y-40), (x, y-80), (x+70, y-40)]
    draw.polygon(points, fill=(139, 69, 19), outline=(0, 0, 0), width=2)
    
    # Door
    draw.rectangle([x-15, y-20, x+15, y+40], fill=(101, 67, 33), outline=(0, 0, 0), width=2)
    
    # Windows
    draw.rectangle([x-45, y-25, x-25, y-5], fill=(135, 206, 235), outline=(0, 0, 0), width=2)
    draw.rectangle([x+25, y-25, x+45, y-5], fill=(135, 206, 235), outline=(0, 0, 0), width=2)

def draw_giramille_tree(draw, x, y, colors):
    """Draw Giramille style tree"""
    trunk_color = (101, 67, 33)
    leaves_color = colors[1] if len(colors) > 1 else (34, 139, 34)
    
    # Trunk
    draw.rectangle([x-8, y-40, x+8, y], fill=trunk_color, outline=(0, 0, 0), width=2)
    
    # Leaves
    draw.ellipse([x-40, y-80, x+40, y-20], fill=leaves_color, outline=(0, 0, 0), width=2)

def draw_giramille_car(draw, x, y, colors):
    """Draw Giramille style car"""
    car_color = colors[0] if colors else (255, 0, 0)
    
    # Car body
    draw.rectangle([x-50, y-20, x+50, y+20], fill=car_color, outline=(0, 0, 0), width=2)
    
    # Wheels
    draw.ellipse([x-40, y+10, x-20, y+30], fill=(0, 0, 0), outline=(0, 0, 0), width=2)
    draw.ellipse([x+20, y+10, x+40, y+30], fill=(0, 0, 0), outline=(0, 0, 0), width=2)

def draw_giramille_person(draw, x, y, colors):
    """Draw Giramille style person"""
    skin_color = (255, 220, 177)
    clothes_color = colors[0] if colors else (0, 0, 255)
    
    # Head
    draw.ellipse([x-15, y-40, x+15, y-10], fill=skin_color, outline=(0, 0, 0), width=2)
    
    # Body
    draw.rectangle([x-20, y-10, x+20, y+30], fill=clothes_color, outline=(0, 0, 0), width=2)
    
    # Arms
    draw.rectangle([x-30, y-5, x-20, y+20], fill=skin_color, outline=(0, 0, 0), width=2)
    draw.rectangle([x+20, y-5, x+30, y+20], fill=skin_color, outline=(0, 0, 0), width=2)
    
    # Legs
    draw.rectangle([x-15, y+30, x-5, y+50], fill=(0, 0, 0), outline=(0, 0, 0), width=2)
    draw.rectangle([x+5, y+30, x+15, y+50], fill=(0, 0, 0), outline=(0, 0, 0), width=2)

def draw_giramille_animal(draw, x, y, colors):
    """Draw Giramille style animal"""
    animal_color = colors[0] if colors else (139, 69, 19)
    
    # Body
    draw.ellipse([x-25, y-15, x+25, y+15], fill=animal_color, outline=(0, 0, 0), width=2)
    
    # Head
    draw.ellipse([x-15, y-35, x+15, y-5], fill=animal_color, outline=(0, 0, 0), width=2)
    
    # Ears
    draw.ellipse([x-20, y-40, x-10, y-30], fill=animal_color, outline=(0, 0, 0), width=2)
    draw.ellipse([x+10, y-40, x+20, y-30], fill=animal_color, outline=(0, 0, 0), width=2)
    
    # Tail
    draw.ellipse([x+20, y-5, x+35, y+10], fill=animal_color, outline=(0, 0, 0), width=2)

def add_giramille_details(draw, width, height, colors):
    """Add Giramille style details"""
    # Sun
    draw.ellipse([width-80, 20, width-20, 80], fill=(255, 255, 0), outline=(0, 0, 0), width=2)
    
    # Clouds
    for i in range(3):
        x = 50 + i * 150
        y = 30 + i * 10
        draw.ellipse([x, y, x+40, y+20], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        draw.ellipse([x+20, y-10, x+60, y+10], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        draw.ellipse([x+40, y, x+80, y+20], fill=(255, 255, 255), outline=(0, 0, 0), width=1)


def draw_giramille_pen(draw, x, y, colors):
    """Draw a simple stylized pen/pencil"""
    body_color = colors[0] if colors else (30, 144, 255)
    tip_color = (200, 160, 0)
    # Pen body (rotated rectangle approximated with polygon)
    w = 100
    h = 14
    # Draw as a slanted rectangle using polygon
    points = [(x - w//2, y - h//2), (x + w//2, y - h//2), (x + w//2 - 10, y + h//2), (x - w//2 - 10, y + h//2)]
    draw.polygon(points, fill=body_color, outline=(0, 0, 0))
    # Tip
    tip = [(x + w//2 - 10, y - 6), (x + w//2 + 8, y), (x + w//2 - 10, y + 6)]
    draw.polygon(tip, fill=tip_color, outline=(0, 0, 0))
    # Accent ring
    ring_x = x - 10
    draw.rectangle([ring_x - 6, y - 8, ring_x + 6, y + 8], fill=(220, 220, 220), outline=(0, 0, 0))

# Image processing helper functions for retouch
def change_house_color(img: Image.Image, new_color: tuple) -> Image.Image:
    """Change house color in the image"""
    # Convert to numpy array for processing
    arr = np.array(img)
    
    # Simple color replacement - find pixels that look like house colors
    # This is a basic implementation - in production, you'd use more sophisticated detection
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            pixel = arr[y, x]
            # Check if pixel is in typical house color range (browns, reds, etc.)
            if (pixel[0] > 100 and pixel[1] < 150 and pixel[2] < 150):  # Reddish colors
                # Blend with new color
                arr[y, x] = [
                    int(pixel[0] * 0.3 + new_color[0] * 0.7),
                    int(pixel[1] * 0.3 + new_color[1] * 0.7),
                    int(pixel[2] * 0.3 + new_color[2] * 0.7)
                ]
    
    return Image.fromarray(arr)

def add_trees_to_image(img: Image.Image) -> Image.Image:
    """Add trees to the image"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Add a few trees
    for i in range(3):
        x = 50 + i * 150
        y = height - 100
        
        # Tree trunk
        draw.rectangle([x-8, y-40, x+8, y], fill=(101, 67, 33), outline=(0, 0, 0), width=2)
        
        # Tree leaves
        draw.ellipse([x-40, y-80, x+40, y-20], fill=(34, 139, 34), outline=(0, 0, 0), width=2)
    
    return img

def add_clouds_to_image(img: Image.Image) -> Image.Image:
    """Add clouds to the image"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Add clouds
    for i in range(4):
        x = 50 + i * 200
        y = 30 + i * 10
        
        # Cloud shape
        draw.ellipse([x, y, x+60, y+30], fill=(255, 255, 255), outline=(200, 200, 200), width=1)
        draw.ellipse([x+20, y-10, x+80, y+20], fill=(255, 255, 255), outline=(200, 200, 200), width=1)
        draw.ellipse([x+40, y, x+100, y+30], fill=(255, 255, 255), outline=(200, 200, 200), width=1)
    
    return img

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Adjust image brightness"""
    arr = np.array(img).astype(np.float32)
    arr = arr * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'scenes_count': len(scene_graphs),
        'model_loaded': model is not None
    })

@app.route('/api/production/health')
def production_health():
    """Production health endpoint"""
    try:
        if not PRODUCTION_AVAILABLE:
            return jsonify({
                'status': 'unavailable',
                'reason': 'Production generator not available on this environment',
                'timestamp': datetime.now().isoformat()
            })
        # Lazy import to avoid heavy startup costs
        try:
            from production_system import initialize_production_system
        except Exception as e:
            return jsonify({'status': 'error', 'error': f'Failed to import production_system: {e}'}), 500
        generator = initialize_production_system()
        health_status = generator.get_health_status()
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/production/metrics')
def production_metrics():
    """Production metrics endpoint"""
    try:
        if not PRODUCTION_AVAILABLE:
            return jsonify({'status': 'unavailable'}), 200
        try:
            from production_system import initialize_production_system
        except Exception as e:
            return jsonify({'status': 'error', 'error': f'Failed to import production_system: {e}'}), 500
        generator = initialize_production_system()
        metrics = generator.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vectorize', methods=['POST'])
def vectorize_api():
    """Vectorize input image to SVG using the in-memory SmallUNet pipeline."""
    if vec_model is None:
        return {"error": "No vectorization model loaded on backend!"}, 500
        
    try:
        # Get input image
        if 'file' in request.files:
            file = request.files['file']
            img = Image.open(file)
        elif request.is_json:
            data = request.get_json()
            if not data.get('image'):
                return {"error": "No image provided in JSON data."}, 400
            img = ImageIO.from_base64(data['image'])
        else:
            return {"error": "No image provided (upload PNG/JPG as file or base64)."}, 400
            
        # Process image through vectorization pipeline
        img = img.convert('RGB').resize((256, 256))
        arr = np.array(img)
        x = torch.from_numpy(arr[:, :, ::-1]).float().permute(2, 0, 1) / 255.0
        x = x.unsqueeze(0).to(device)
        
        # Generate mask
        with torch.no_grad():
            logits = vec_model(x)
            mask = logits.argmax(dim=1)[0].byte().cpu().numpy()
            
        # Process paths
        layer_paths = []
        for cls in range(num_classes):
            cls_mask = (mask == cls).astype(np.uint8) * 255
            if cls_mask.sum() < 10:
                continue
            paths = contours_to_beziers(cls_mask, epsilon=1.5, max_segments=8)
            paths = reduce_anchors(paths, max_anchors=300)
            layer_paths.append(paths)
            
        # Generate SVG
        merged = merge_layers(layer_paths, max_layers=20)
        svg_buffer = ImageIO.create_buffer()
        beziers_to_svg(merged, svg_buffer, size=(256, 256))
        svg_buffer.seek(0)
        
        return send_file(
            svg_buffer,
            mimetype="image/svg+xml",
            as_attachment=True,
            download_name="vectorized.svg"
        )
    except Exception as e:
        return {"error": str(e)}, 500


# --- Added endpoints for Stage2 UI actions ---
@app.route('/api/retouch', methods=['POST'])
def retouch_api():
    data = request.get_json()
    image_b64 = data.get('image')
    prompt = data.get('prompt', '')
    
    if not image_b64:
        return {"error": "No image provided."}, 400
    
    try:
        # Load and process image
        img = ImageIO.from_base64(image_b64).convert('RGB')
        
        # Apply different effects based on prompt
        if prompt:
            prompt_lower = prompt.lower()
            
            # Color changes
            if 'brown' in prompt_lower and 'house' in prompt_lower:
                # Change house color to brown
                img = change_house_color(img, (139, 69, 19))  # Brown color
            elif 'blue' in prompt_lower and 'house' in prompt_lower:
                # Change house color to blue
                img = change_house_color(img, (0, 100, 200))
            elif 'red' in prompt_lower and 'house' in prompt_lower:
                # Change house color to red
                img = change_house_color(img, (200, 0, 0))
            
            # Add trees
            if 'tree' in prompt_lower or 'trees' in prompt_lower:
                img = add_trees_to_image(img)
            
            # Add clouds
            if 'cloud' in prompt_lower or 'sky' in prompt_lower:
                img = add_clouds_to_image(img)
            
            # Brightness adjustment
            if 'bright' in prompt_lower or 'brighter' in prompt_lower:
                img = adjust_brightness(img, 1.3)
            elif 'dark' in prompt_lower or 'darker' in prompt_lower:
                img = adjust_brightness(img, 0.7)
            
            # Blur effect for "soft" or "smooth"
            if 'soft' in prompt_lower or 'smooth' in prompt_lower:
                arr = cv2.GaussianBlur(np.array(img), (15, 15), 0)
                img = Image.fromarray(arr)
            else:
                # Default subtle enhancement
                arr = cv2.GaussianBlur(np.array(img), (3, 3), 0)
                img = Image.fromarray(arr)
        else:
            # Default processing
            arr = cv2.GaussianBlur(np.array(img), (7, 7), 0)
            img = Image.fromarray(arr)
        
        # Convert back to base64
        img_b64 = ImageIO.to_base64(img)
        return jsonify({"image": f"data:image/png;base64,{img_b64}"})
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/resize', methods=['POST'])
def resize_api():
    data = request.get_json()
    image_b64 = data.get('image')
    width = data.get('width', 256)
    height = data.get('height', 256)
    if not image_b64:
        return {"error": "No image provided."}, 400
    try:
        # Load and resize image
        img = ImageIO.from_base64(image_b64).convert('RGB')
        img = img.resize((width, height))
        
        # Convert back to base64
        img_b64 = ImageIO.to_base64(img)
        return jsonify({"image": f"data:image/png;base64,{img_b64}"})
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/positions', methods=['POST'])
def positions_api():
    data = request.get_json()
    image_b64 = data.get('image')
    if not image_b64:
        return {"error": "No image provided."}, 400
    try:
        # Convert base64 to image, process, and convert back
        img = ImageIO.from_base64(image_b64).convert('RGB')
        # Dummy: return same image, but could add object detection here
        b64_data = ImageIO.to_base64(img)
        return jsonify({"image": f"data:image/png;base64,{b64_data}"})
    except Exception as e:
        return {"error": str(e)}, 500

try:
    # Backwards-compatibility mappings for frontends that call endpoints without the /api prefix
    app.add_url_rule('/generate', view_func=generate_image, methods=['POST'])
    app.add_url_rule('/vectorize', view_func=vectorize_api, methods=['POST'])
    app.add_url_rule('/multiview/generate', view_func=generate_multiview, methods=['POST'])
    app.add_url_rule('/upload', view_func=upload_asset, methods=['POST'])
    app.add_url_rule('/export/scene', view_func=export_scene, methods=['POST'])
    app.add_url_rule('/health', view_func=health_check, methods=['GET'])
except NameError:
    # If the functions are not defined yet (file being edited), skip mapping.
    pass

if __name__ == '__main__':
    # Run without debugger/reloader in automated test environment
    app.run(debug=False, host='0.0.0.0', port=5000)
