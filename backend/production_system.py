"""
Production-Ready Giramille Style System
Optimized for production deployment with caching, monitoring, and error handling
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import redis
import psutil
import threading
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionConfig:
    """Production configuration"""
    
    def __init__(self):
        self.max_concurrent_requests = 10
        self.cache_ttl = 3600  # 1 hour
        self.max_image_size = (1024, 1024)
        self.default_image_size = (512, 512)
        self.quality_settings = {
            'fast': {'steps': 20, 'guidance': 6.0},
            'balanced': {'steps': 35, 'guidance': 7.0},
            'high': {'steps': 50, 'guidance': 7.5}
        }
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.model_cache_dir = 'models/cache'
        self.output_dir = 'outputs/production'

class ImageCache:
    """Redis-based image cache for production"""
    
    def __init__(self, redis_url: str):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("[SUCCESS] Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available, using memory cache: {e}")
            self.redis_client = None
            self.memory_cache = {}
    
    def get(self, key: str) -> Optional[bytes]:
        """Get image from cache"""
        try:
            if self.redis_client:
                return self.redis_client.get(key)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: int = 3600):
        """Set image in cache"""
        try:
            if self.redis_client:
                self.redis_client.setex(key, ttl, value)
            else:
                self.memory_cache[key] = value
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def generate_key(self, prompt: str, style: str, quality: str) -> str:
        """Generate cache key"""
        content = f"{prompt}_{style}_{quality}"
        return hashlib.md5(content.encode()).hexdigest()

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'avg_generation_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.lock = threading.Lock()
    
    def record_request(self, success: bool, generation_time: float, cache_hit: bool):
        """Record request metrics"""
        with self.lock:
            self.metrics['requests_total'] += 1
            if success:
                self.metrics['requests_success'] += 1
            else:
                self.metrics['requests_failed'] += 1
            
            if cache_hit:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1
            
            # Update average generation time
            total_success = self.metrics['requests_success']
            if total_success > 0:
                current_avg = self.metrics['avg_generation_time']
                self.metrics['avg_generation_time'] = (current_avg * (total_success - 1) + generation_time) / total_success
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }

class ProductionGiramilleGenerator:
    """Production-ready Giramille style generator"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.cache = ImageCache(config.redis_url)
        self.monitor = PerformanceMonitor()
        self.request_queue = Queue(maxsize=config.max_concurrent_requests)
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        
        # Initialize
        self._setup_directories()
        self._load_model()
    
    def _setup_directories(self):
        """Setup required directories"""
        os.makedirs(self.config.model_cache_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def _load_model(self):
        """Load the production model"""
        try:
            logger.info("Loading production model...")
            
            # Try to load trained model first
            trained_model_path = os.path.join(self.config.model_cache_dir, "giramille_style_model")
            if os.path.exists(trained_model_path):
                logger.info("Loading trained Giramille model...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    trained_model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                # Fallback to base model
                logger.info("Loading base model...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            # Memory efficient attention may not be available in all diffusers versions
            try:
                self.pipe.enable_memory_efficient_attention()
            except AttributeError:
                logger.info("Memory efficient attention not available, using standard attention")
            
            logger.info("[SUCCESS] Production model loaded successfully!")
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading model: {e}")
            raise
    
    def generate_image(self, prompt: str, style: str = "png", quality: str = "balanced") -> Dict:
        """Generate image with production optimizations"""
        start_time = time.time()
        cache_hit = False
        
        try:
            # Check cache first
            cache_key = self.cache.generate_key(prompt, style, quality)
            cached_image = self.cache.get(cache_key)
            
            if cached_image:
                logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                cache_hit = True
                generation_time = time.time() - start_time
                self.monitor.record_request(True, generation_time, True)
                
                return {
                    'success': True,
                    'image': cached_image,
                    'cached': True,
                    'generation_time': generation_time,
                    'prompt': prompt,
                    'style': style,
                    'quality': quality
                }
            
            # Generate new image
            logger.info(f"Generating new image for prompt: {prompt[:50]}...")
            
            # Enhance prompt
            enhanced_prompt = self._enhance_prompt_for_production(prompt, quality)
            
            # Get quality settings (matching training quality)
            quality_settings = self.config.quality_settings.get(quality, self.config.quality_settings['balanced'])
            
            # Use higher quality settings for better results
            if quality == 'high':
                quality_settings = {'steps': 50, 'guidance': 7.5}
            elif quality == 'balanced':
                quality_settings = {'steps': 35, 'guidance': 7.0}
            else:  # fast
                quality_settings = {'steps': 25, 'guidance': 6.5}
                
            # Generate image
            with torch.no_grad():
                result = self.pipe(
                    enhanced_prompt,
                    num_inference_steps=quality_settings['steps'],
                    guidance_scale=quality_settings['guidance'],
                    height=self.config.default_image_size[0],
                    width=self.config.default_image_size[1],
                    generator=torch.Generator().manual_seed(42)
                )
                
                image = result.images[0]
                
                # Apply Giramille style post-processing
                image = self._apply_production_style(image, prompt)
                
                # Convert to bytes
                import io
                buffer = io.BytesIO()
                image.save(buffer, format='PNG', optimize=True)
                image_bytes = buffer.getvalue()
                
                # Cache the result
                self.cache.set(cache_key, image_bytes, self.config.cache_ttl)
                
                generation_time = time.time() - start_time
                self.monitor.record_request(True, generation_time, False)
                
                logger.info(f"[SUCCESS] Generated image in {generation_time:.2f}s")
                
                return {
                    'success': True,
                    'image': image_bytes,
                    'cached': False,
                    'generation_time': generation_time,
                    'prompt': prompt,
                    'style': style,
                    'quality': quality
                }
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.monitor.record_request(False, generation_time, cache_hit)
            logger.error(f"[ERROR] Error generating image: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'generation_time': generation_time,
                'prompt': prompt,
                'style': style,
                'quality': quality
            }
    
    def _enhance_prompt_for_production(self, prompt: str, quality: str) -> str:
        """Enhance prompt for production with quality-based optimization"""
        
        # Base Giramille style
        giramille_style = "cartoon style, colorful, vibrant, cute, friendly, playful, vector art style, flat design, modern illustration"
        
        # Quality-based enhancements (matching training quality)
        quality_enhancements = {
            'fast': "cartoon style, colorful, vibrant, cute, friendly, playful, vector art style, flat design, modern illustration, simple, clean",
            'balanced': "cartoon style, colorful, vibrant, cute, friendly, playful, vector art style, flat design, modern illustration, detailed, professional, high quality",
            'high': "cartoon style, colorful, vibrant, cute, friendly, playful, vector art style, flat design, modern illustration, ultra detailed, professional, masterpiece, high resolution, perfect composition, Giramille style"
        }
        
        quality_text = quality_enhancements.get(quality, quality_enhancements['balanced'])
        
        # Color accuracy enhancement
        color_enhanced = self._enhance_color_accuracy(prompt)
        
        # Combine all enhancements
        enhanced = f"{color_enhanced}, {giramille_style}, {quality_text}, accurate colors"
        
        return enhanced
    
    def _enhance_color_accuracy(self, prompt: str) -> str:
        """Enhanced color accuracy for production"""
        prompt_lower = prompt.lower()
        
        # Advanced color mappings
        color_enhancements = {
            'blue house': 'bright blue house, vivid blue building, blue colored house, azure house',
            'red house': 'bright red house, vivid red building, red colored house, crimson house',
            'green house': 'bright green house, vivid green building, green colored house, emerald house',
            'yellow house': 'bright yellow house, vivid yellow building, yellow colored house, golden house',
            'purple house': 'bright purple house, vivid purple building, purple colored house, violet house',
            'orange house': 'bright orange house, vivid orange building, orange colored house, amber house',
            'pink house': 'bright pink house, vivid pink building, pink colored house, rose house',
            'blue car': 'bright blue car, vivid blue vehicle, blue colored car, azure car',
            'red car': 'bright red car, vivid red vehicle, red colored car, crimson car',
            'green car': 'bright green car, vivid green vehicle, green colored car, emerald car',
            'green horse': 'bright green horse, vivid green animal, green colored horse, emerald horse',
            'yellow car': 'bright yellow car, vivid yellow vehicle, yellow colored car, golden car',
            'purple mountain': 'bright purple mountain, vivid purple peak, purple colored mountain, violet mountain',
            'orange sunset': 'bright orange sunset, vivid orange sky, orange colored sunset, amber sunset'
        }
        
        enhanced_prompt = prompt
        for original, enhanced in color_enhancements.items():
            if original in prompt_lower:
                enhanced_prompt = enhanced_prompt.replace(original, enhanced)
        
        return enhanced_prompt
    
    def _apply_production_style(self, image: Image.Image, prompt: str) -> Image.Image:
        """Apply production-optimized Giramille style"""
        
        # Convert to RGBA for better processing
        image = image.convert('RGBA')
        
        # Enhance colors for Giramille style
        image = self._enhance_colors_production(image, prompt)
        
        # Add Giramille style elements
        image = self._add_giramille_elements_production(image, prompt)
        
        return image
    
    def _enhance_colors_production(self, image: Image.Image, prompt: str) -> Image.Image:
        """Production-optimized color enhancement"""
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Enhance saturation and brightness
        img_array = img_array.astype(np.float32)
        
        # Increase saturation
        img_array[:, :, :3] = np.clip(img_array[:, :, :3] * 1.15, 0, 255)
        
        # Slight brightness increase
        img_array[:, :, :3] = np.clip(img_array[:, :, :3] + 8, 0, 255)
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(img_array.astype(np.uint8))
        
        return enhanced_image
    
    def _add_giramille_elements_production(self, image: Image.Image, prompt: str) -> Image.Image:
        """Add production-optimized Giramille elements"""
        
        # Create overlay for decorative elements
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Add subtle decorative elements
        if 'house' in prompt.lower():
            self._add_decorative_elements_production(draw, image.size)
        
        # Blend overlay with original image
        image = Image.alpha_composite(image, overlay)
        
        return image
    
    def _add_decorative_elements_production(self, draw: ImageDraw.Draw, size: tuple):
        """Add production-optimized decorative elements"""
        width, height = size
        
        # Add subtle decorative elements
        for i in range(3):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(3, 8)
            color = (255, 255, 255, 20)  # Very subtle
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    def get_health_status(self) -> Dict:
        """Get system health status"""
        metrics = self.monitor.get_metrics()
        system_info = self.monitor.get_system_info()
        
        return {
            'status': 'healthy' if system_info['cpu_percent'] < 90 and system_info['memory_percent'] < 90 else 'warning',
            'metrics': metrics,
            'system': system_info,
            'model_loaded': self.pipe is not None,
            'cache_available': self.cache.redis_client is not None,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict:
        """Get production metrics"""
        return self.monitor.get_metrics()
    
    def clear_cache(self):
        """Clear image cache"""
        try:
            if self.cache.redis_client:
                self.cache.redis_client.flushdb()
                logger.info("[SUCCESS] Cache cleared")
            else:
                self.cache.memory_cache.clear()
                logger.info("[SUCCESS] Memory cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

# Global production instance
production_generator = None

def initialize_production_system():
    """Initialize production system"""
    global production_generator
    
    if production_generator is None:
        config = ProductionConfig()
        production_generator = ProductionGiramilleGenerator(config)
        logger.info("[STARTUP] Production system initialized")
    
    return production_generator

def generate_production_image(prompt: str, style: str = "png", quality: str = "balanced") -> Dict:
    """Generate image using production system"""
    generator = initialize_production_system()
    return generator.generate_image(prompt, style, quality)

if __name__ == "__main__":
    # Test production system
    logger.info("🧪 Testing production system...")
    
    generator = initialize_production_system()
    
    # Test prompts
    test_prompts = [
        "blue house with red car, cartoon style",
        "green tree with yellow flowers, nature scene",
        "purple mountain with orange sunset, landscape"
    ]
    
    for prompt in test_prompts:
        result = generator.generate_image(prompt, quality="balanced")
        if result['success']:
            logger.info(f"[SUCCESS] Generated: {prompt} in {result['generation_time']:.2f}s")
        else:
            logger.error(f"[ERROR] Failed: {prompt} - {result['error']}")
    
    # Show metrics
    metrics = generator.get_metrics()
    logger.info(f"[METRICS] {metrics}")
    
    # Show health status
    health = generator.get_health_status()
    logger.info(f"[HEALTH] {health['status']}")
