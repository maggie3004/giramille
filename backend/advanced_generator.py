"""
Advanced AI Image Generator for Giramille Style
This will create proper AI-generated images like Gemini/Freepik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
import io
import base64
from typing import List, Dict, Optional
import json
import os

class GiramilleStyleGenerator:
    """Advanced AI generator for Giramille style images"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.style_embeddings = {}
        self.load_model()
    
    def load_model(self):
        """Load the AI model"""
        try:
            # Load Stable Diffusion for high-quality generation
            model_id = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            print("[SUCCESS] AI Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            self.pipe = None
    
    def generate_giramille_image(self, prompt: str, style: str = "png") -> Image.Image:
        """Generate high-quality Giramille style image"""
        
        if not self.pipe:
            return self._fallback_generation(prompt, style)
        
        try:
            # Enhance prompt for Giramille style
            enhanced_prompt = self._enhance_prompt_for_giramille(prompt)
            
            # Generate with AI
            with torch.no_grad():
                result = self.pipe(
                    enhanced_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator().manual_seed(42)
                )
            
            image = result.images[0]
            
            # Apply Giramille style post-processing
            image = self._apply_giramille_style(image, prompt)
            
            return image
            
        except Exception as e:
            print(f"[ERROR] AI generation failed: {e}")
            return self._fallback_generation(prompt, style)
    
    def _enhance_prompt_for_giramille(self, prompt: str) -> str:
        """Enhance prompt for better Giramille style generation with improved color accuracy"""
        
        # Base Giramille style keywords
        giramille_style = "cartoon style, colorful, vibrant, cute, friendly, playful, vector art style, flat design, modern illustration"
        
        # Enhanced color processing for better accuracy
        color_enhanced_prompt = self._enhance_color_accuracy(prompt)
        
        # Object enhancement
        objects = self._detect_objects_in_prompt(prompt)
        object_text = ", ".join(objects) if objects else "scene"
        
        # Combine everything with emphasis on color accuracy
        enhanced = f"{color_enhanced_prompt}, {object_text}, {giramille_style}, high quality, detailed, professional illustration, accurate colors"
        
        return enhanced
    
    def _enhance_color_accuracy(self, prompt: str) -> str:
        """Enhance color accuracy in prompts"""
        prompt_lower = prompt.lower()
        
        # Color mapping with stronger emphasis
        color_enhancements = {
            'blue house': 'bright blue house, vivid blue building, blue colored house',
            'red house': 'bright red house, vivid red building, red colored house',
            'green house': 'bright green house, vivid green building, green colored house',
            'yellow house': 'bright yellow house, vivid yellow building, yellow colored house',
            'purple house': 'bright purple house, vivid purple building, purple colored house',
            'orange house': 'bright orange house, vivid orange building, orange colored house',
            'pink house': 'bright pink house, vivid pink building, pink colored house',
            'blue car': 'bright blue car, vivid blue vehicle, blue colored car',
            'red car': 'bright red car, vivid red vehicle, red colored car',
            'green car': 'bright green car, vivid green vehicle, green colored car',
            'yellow car': 'bright yellow car, vivid yellow vehicle, yellow colored car',
            'blue tree': 'bright blue tree, vivid blue plant, blue colored tree',
            'green tree': 'bright green tree, vivid green plant, green colored tree',
            'green horse': 'bright green horse, vivid green animal, green colored horse, emerald horse',
            'purple mountain': 'bright purple mountain, vivid purple peak, purple colored mountain',
            'orange sunset': 'bright orange sunset, vivid orange sky, orange colored sunset'
        }
        
        # Apply color enhancements
        enhanced_prompt = prompt
        for original, enhanced in color_enhancements.items():
            if original in prompt_lower:
                enhanced_prompt = enhanced_prompt.replace(original, enhanced)
        
        return enhanced_prompt
    
    def _extract_colors_from_prompt(self, prompt: str) -> List[str]:
        """Extract and enhance colors from prompt"""
        colors = []
        prompt_lower = prompt.lower()
        
        color_map = {
            'red': 'bright red', 'blue': 'vibrant blue', 'green': 'vibrant green, bright green, vivid green, emerald green',
            'yellow': 'sunny yellow', 'purple': 'royal purple', 'orange': 'warm orange',
            'pink': 'soft pink', 'brown': 'rich brown', 'black': 'deep black',
            'white': 'pure white', 'gray': 'elegant gray', 'cyan': 'bright cyan'
        }
        
        for color_name, enhanced_color in color_map.items():
            if color_name in prompt_lower:
                colors.append(enhanced_color)
        
        return colors
    
    def _detect_objects_in_prompt(self, prompt: str) -> List[str]:
        """Detect and enhance objects from prompt"""
        objects = []
        prompt_lower = prompt.lower()
        
        object_enhancements = {
            'house': 'beautiful house, home, building',
            'car': 'cute car, vehicle, automobile',
            'tree': 'green tree, nature, plant',
            'person': 'friendly person, character, people',
            'animal': 'cute animal, pet, creature',
            'dog': 'adorable dog, puppy, pet',
            'cat': 'cute cat, kitten, feline',
            'horse': 'beautiful horse, galloping horse, equine animal, running horse',
            'flower': 'beautiful flower, bloom, garden',
            'mountain': 'majestic mountain, landscape, nature',
            'river': 'flowing river, water, stream',
            'sun': 'bright sun, sunshine, daylight',
            'moon': 'glowing moon, night sky, celestial'
        }
        
        for obj, enhancement in object_enhancements.items():
            if obj in prompt_lower:
                objects.append(enhancement)
        
        return objects
    
    def _apply_giramille_style(self, image: Image.Image, prompt: str) -> Image.Image:
        """Apply Giramille style post-processing"""
        
        # Convert to RGBA for better color manipulation
        image = image.convert('RGBA')
        
        # Enhance colors for Giramille style
        image = self._enhance_colors(image, prompt)
        
        # Add Giramille style elements
        image = self._add_giramille_elements(image, prompt)
        
        return image
    
    def _enhance_colors(self, image: Image.Image, prompt: str) -> Image.Image:
        """Enhance colors for Giramille style"""
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Increase saturation
        img_array = img_array.astype(np.float32)
        img_array[:, :, :3] = np.clip(img_array[:, :, :3] * 1.2, 0, 255)
        
        # Increase brightness slightly
        img_array[:, :, :3] = np.clip(img_array[:, :, :3] + 10, 0, 255)
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(img_array.astype(np.uint8))
        
        return enhanced_image
    
    def _add_giramille_elements(self, image: Image.Image, prompt: str) -> Image.Image:
        """Add Giramille style decorative elements"""
        
        # Create overlay for decorative elements
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Add some decorative elements based on prompt
        if 'house' in prompt.lower():
            # Add some decorative elements around house
            self._add_decorative_elements(draw, image.size)
        
        # Blend overlay with original image
        image = Image.alpha_composite(image, overlay)
        
        return image
    
    def _add_decorative_elements(self, draw: ImageDraw.Draw, size: tuple):
        """Add decorative elements"""
        width, height = size
        
        # Add some decorative circles
        for i in range(5):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(5, 15)
            color = (255, 255, 255, 30)  # Semi-transparent white
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    def _fallback_generation(self, prompt: str, style: str) -> Image.Image:
        """Fallback generation when AI model is not available"""
        
        # Create a high-quality fallback image
        width, height = 512, 512
        image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Parse prompt
        colors = self._extract_colors_from_prompt(prompt)
        objects = self._detect_objects_in_prompt(prompt)
        
        # Create background
        bg_color = (135, 206, 235)  # Sky blue
        if colors:
            # Use first detected color
            color_name = colors[0].split()[0]  # Get base color name
            color_map = {
                'red': (255, 99, 99), 'blue': (99, 99, 255), 'green': (99, 255, 99),
                'yellow': (255, 255, 99), 'purple': (255, 99, 255), 'orange': (255, 165, 99)
            }
            bg_color = color_map.get(color_name, (135, 206, 235))
        
        draw.rectangle([0, 0, width, height], fill=bg_color)
        
        # Draw objects based on prompt
        if 'house' in objects or 'home' in objects:
            self._draw_advanced_house(draw, width//2, height//2, colors)
        if 'car' in objects or 'vehicle' in objects:
            self._draw_advanced_car(draw, 400, height-150, colors)
        if 'tree' in objects or 'forest' in objects:
            self._draw_advanced_tree(draw, 100, height-100, colors)
        
        return image
    
    def _draw_advanced_house(self, draw: ImageDraw.Draw, x: int, y: int, colors: List[str]):
        """Draw an advanced house"""
        house_color = (255, 182, 193)  # Pink
        if colors:
            color_name = colors[0].split()[0]
            color_map = {
                'red': (255, 99, 99), 'blue': (99, 99, 255), 'green': (99, 255, 99),
                'yellow': (255, 255, 99), 'purple': (255, 99, 255), 'orange': (255, 165, 99)
            }
            house_color = color_map.get(color_name, (255, 182, 193))
        
        # House body
        draw.rectangle([x-60, y-40, x+60, y+40], fill=house_color, outline=(0, 0, 0, 255), width=2)
        
        # Roof
        roof_points = [(x-70, y-40), (x, y-80), (x+70, y-40)]
        draw.polygon(roof_points, fill=(139, 69, 19), outline=(0, 0, 0, 255), width=2)
        
        # Door
        draw.rectangle([x-15, y-10, x+15, y+40], fill=(101, 67, 33), outline=(0, 0, 0, 255), width=2)
        
        # Windows
        draw.rectangle([x-45, y-20, x-25, y-5], fill=(173, 216, 230), outline=(0, 0, 0, 255), width=2)
        draw.rectangle([x+25, y-20, x+45, y-5], fill=(173, 216, 230), outline=(0, 0, 0, 255), width=2)
    
    def _draw_advanced_car(self, draw: ImageDraw.Draw, x: int, y: int, colors: List[str]):
        """Draw an advanced car"""
        car_color = (255, 0, 0)  # Red
        if colors:
            color_name = colors[0].split()[0]
            color_map = {
                'red': (255, 99, 99), 'blue': (99, 99, 255), 'green': (99, 255, 99),
                'yellow': (255, 255, 99), 'purple': (255, 99, 255), 'orange': (255, 165, 99)
            }
            car_color = color_map.get(color_name, (255, 0, 0))
        
        # Car body
        draw.rectangle([x-50, y-20, x+50, y+20], fill=car_color, outline=(0, 0, 0, 255), width=2)
        
        # Wheels
        draw.ellipse([x-40, y+10, x-20, y+30], fill=(50, 50, 50), outline=(0, 0, 0, 255), width=2)
        draw.ellipse([x+20, y+10, x+40, y+30], fill=(50, 50, 50), outline=(0, 0, 0, 255), width=2)
        
        # Windshield
        draw.rectangle([x-30, y-15, x+30, y-5], fill=(173, 216, 230), outline=(0, 0, 0, 255), width=1)
    
    def _draw_advanced_tree(self, draw: ImageDraw.Draw, x: int, y: int, colors: List[str]):
        """Draw an advanced tree"""
        # Trunk
        draw.rectangle([x-5, y-20, x+5, y+20], fill=(101, 67, 33), outline=(0, 0, 0, 255), width=2)
        
        # Leaves
        leaf_color = (34, 139, 34)  # Green
        if colors:
            color_name = colors[0].split()[0]
            if color_name == 'green':
                leaf_color = (34, 139, 34)
        
        draw.ellipse([x-30, y-50, x+30, y-10], fill=leaf_color, outline=(0, 0, 0, 255), width=2)

# Global instance
giramille_generator = GiramilleStyleGenerator()

def generate_giramille_image_advanced(prompt: str, style: str = "png") -> Image.Image:
    """Generate advanced Giramille style image"""
    return giramille_generator.generate_giramille_image(prompt, style)
