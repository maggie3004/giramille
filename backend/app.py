from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from typing import Dict, List, Any
import uuid
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
"""Optional heavy dependencies (diffusers/xformers) are guarded.
If unavailable on CPU-only or incompatible Python, we fall back to local generation.
"""
PRODUCTION_AVAILABLE = False
try:
    from advanced_generator import generate_giramille_image_advanced  # noqa: F401
    from production_system import generate_production_image, initialize_production_system  # noqa: F401
    PRODUCTION_AVAILABLE = True
except Exception as _optional_import_error:
    # Keep running with lightweight generator
    PRODUCTION_AVAILABLE = False

app = Flask(__name__)
CORS(app)

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

def load_giramille_model():
    global model
    try:
        model = GiramilleStyleEncoder(num_classes=4)
        checkpoint_path = 'models/giramille_best_epoch_31_acc_74.1.pth'
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.eval()
                model.to(device)
                print(f"✅ Giramille model loaded successfully! Accuracy: {checkpoint.get('accuracy', 'Unknown')}")
            except Exception as load_error:
                print(f"⚠️ Model architecture mismatch, using random weights: {load_error}")
                model.eval()
                model.to(device)
        else:
            print("⚠️ No trained model found, using random weights")
            model.eval()
            model.to(device)
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Create a simple fallback model
        model = GiramilleStyleEncoder(num_classes=4)
        model.eval()
        model.to(device)
        return True

# Load model on startup
load_giramille_model()

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
        # Decode base64 image
        image_data = base64.b64decode(source_image.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Simulate AI generation (in production, this would call your AI model)
        generated_image = simulate_multiview_generation(image, target_angle)
        
        # Convert to base64
        buffer = io.BytesIO()
        generated_image.save(buffer, format='PNG')
        buffer.seek(0)
        generated_b64 = base64.b64encode(buffer.getvalue()).decode()
        
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
                img = Image.open(io.BytesIO(image_data))
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

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """Generate Giramille style image from prompt"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        style = data.get('style', 'png')  # png or vector
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Prefer production system if available, otherwise fallback to local generator
        generated_image = None
        if PRODUCTION_AVAILABLE:
            try:
                quality = data.get('quality', 'balanced')  # fast, balanced, high
                result = generate_production_image(prompt, style, quality)
                if result.get('success'):
                    import io
                    generated_image = Image.open(io.BytesIO(result['image']))
                else:
                    # Fall back if production failed
                    generated_image = generate_giramille_image(prompt, style)
            except Exception:
                generated_image = generate_giramille_image(prompt, style)
        else:
            generated_image = generate_giramille_image(prompt, style)
        
        # Convert to base64
        buffer = io.BytesIO()
        generated_image.save(buffer, format='PNG')
        buffer.seek(0)
        generated_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{generated_b64}",
            'prompt': prompt,
            'style': style,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_giramille_image(prompt: str, style: str) -> Image.Image:
    """Generate Giramille style image from prompt"""
    # Create canvas
    width, height = 512, 512
    image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Parse prompt for colors and objects
    colors = extract_colors_from_prompt(prompt)
    objects = detect_objects_in_prompt(prompt)
    
    # Generate background with Giramille style
    bg_color = colors[0] if colors else (135, 206, 235)  # Sky blue default
    draw.rectangle([0, 0, width, height], fill=bg_color)
    
    # Add Giramille style elements
    if 'house' in objects or 'home' in objects:
        draw_giramille_house(draw, width//2, height//2, colors)
    if 'tree' in objects or 'forest' in objects:
        draw_giramille_tree(draw, 100, height-100, colors)
    if 'car' in objects or 'vehicle' in objects:
        draw_giramille_car(draw, 400, height-150, colors)
    if 'person' in objects or 'people' in objects:
        draw_giramille_person(draw, 200, height-200, colors)
    if 'animal' in objects or 'dog' in objects or 'cat' in objects:
        draw_giramille_animal(draw, 350, height-180, colors)
    
    # Add Giramille style details
    add_giramille_details(draw, width, height, colors)
    
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
        'animal', 'dog', 'cat', 'bird', 'flower', 'mountain', 'river', 'sun', 'moon', 'star'
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
        generator = initialize_production_system()
        metrics = generator.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
