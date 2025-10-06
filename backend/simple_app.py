"""
Simplified Flask Backend for Giramille AI System
This version works without xformers and flash-attn dependencies
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import json
import time

app = Flask(__name__)
CORS(app)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

class SimpleGiramilleGenerator:
    """Simplified Giramille style generator without heavy dependencies"""
    
    def __init__(self):
        self.style_presets = {
            'cartoon': {
                'description': 'Bright, colorful cartoon style',
                'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            },
            'educational': {
                'description': 'Clean, educational illustration style',
                'colors': ['#74B9FF', '#0984E3', '#00B894', '#FDCB6E', '#E17055']
            },
            'playful': {
                'description': 'Fun, playful character style',
                'colors': ['#FD79A8', '#FDCB6E', '#6C5CE7', '#A29BFE', '#00B894']
            }
        }
    
    def generate_placeholder_image(self, prompt, style='cartoon', width=512, height=512):
        """Generate a placeholder image with Giramille style elements"""
        # Create a base image
        img = Image.new('RGB', (width, height), color='#FFFFFF')
        draw = ImageDraw.Draw(img)
        
        # Get style colors
        colors = self.style_presets.get(style, self.style_presets['cartoon'])['colors']
        
        # Draw some geometric shapes to represent the style
        for i in range(5):
            x = (i * width // 5) + 50
            y = height // 2 - 50
            size = 80
            
            # Draw circle
            draw.ellipse([x, y, x + size, y + size], fill=colors[i % len(colors)])
            
            # Draw some lines
            draw.line([x + size//2, y + size, x + size//2, y + size + 50], 
                     fill=colors[(i + 1) % len(colors)], width=3)
        
        # Add text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
            
        # Draw prompt text
        text_lines = [prompt[i:i+30] for i in range(0, len(prompt), 30)]
        for i, line in enumerate(text_lines[:3]):  # Max 3 lines
            draw.text((50, 50 + i * 20), line, fill='#333333', font=font)
        
        # Add Giramille branding
        draw.text((width - 150, height - 30), "Giramille AI", fill='#666666', font=font)
        
        return img
    
    def enhance_prompt(self, prompt, style):
        """Enhance the prompt with style-specific keywords"""
        style_keywords = {
            'cartoon': 'cartoon, colorful, bright, cheerful, animated style',
            'educational': 'educational, clean, simple, clear, learning illustration',
            'playful': 'playful, fun, cute, adorable, child-friendly'
        }
        
        enhanced = f"{prompt}, {style_keywords.get(style, '')}, high quality, detailed"
        return enhanced

# Initialize the generator
generator = SimpleGiramilleGenerator()

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Giramille AI Backend Server',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            '/generate': 'POST - Generate image',
            '/styles': 'GET - Get available styles',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'generator': 'SimpleGiramilleGenerator'
    })

@app.route('/styles')
def get_styles():
    """Get available art styles"""
    return jsonify({
        'styles': generator.style_presets,
        'default': 'cartoon'
    })

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate an image based on prompt and style"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        prompt = data.get('prompt', '')
        style = data.get('style', 'cartoon')
        width = data.get('width', 512)
        height = data.get('height', 512)
        quality = data.get('quality', 'balanced')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Enhance the prompt
        enhanced_prompt = generator.enhance_prompt(prompt, style)
        
        # Generate the image
        start_time = time.time()
        image = generator.generate_placeholder_image(enhanced_prompt, style, width, height)
        generation_time = time.time() - start_time
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'style': style,
            'generation_time': generation_time,
            'dimensions': {'width': width, 'height': height},
            'quality': quality
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the file
        filename = f"upload_{int(time.time())}_{file.filename}"
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Giramille AI Backend Server...")
    print("Note: This is a simplified version without heavy AI dependencies")
    print("Server will be available at: http://localhost:5000")
    print("Available endpoints:")
    print("   - GET  / : Home")
    print("   - GET  /health : Health check")
    print("   - GET  /styles : Available styles")
    print("   - POST /generate : Generate image")
    print("   - POST /upload : Upload file")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
