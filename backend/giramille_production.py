__all__ = ['ProductionConfig', 'initialize_production_system']
"""
Production-Ready Giramille Style System
Optimized for production deployment with caching, monitoring, and error handling
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw
import numpy as np
import os
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import redis
import psutil
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO
import io
import warnings
# silence only the specific FutureWarning from transformers about _register_pytree_node
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*", module="transformers.*")

# Configure logging
logger = logging.getLogger(__name__)

# Global production instance
production_generator = None

def initialize_production_system(device_type="cpu"):
    """Initialize global production system"""
    global production_generator
    if production_generator is None:
        try:
            config = ProductionConfig(device_type)
            logger.info(f"[STARTUP] Creating production system with {device_type} configuration")
            production_generator = ProductionGiramilleGenerator(config)
            os.makedirs('logs', exist_ok=True)
            logger.info("[STARTUP] Production system initialized successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize production system: {e}")
            raise
    return production_generator

class ProductionConfig:
    """Production configuration"""
    def __init__(self, device_type="cpu"):
        self.device_type = device_type
        self.max_concurrent_requests = 10
        self.cache_ttl = 3600  # 1 hour
        if device_type == "cpu":
            self.max_image_size = (768, 768)
            self.default_image_size = (512, 512)
            self.quality_settings = {
                'fast': {'steps': 20, 'guidance': 7.0},
                'balanced': {'steps': 30, 'guidance': 7.5},
                'high': {'steps': 40, 'guidance': 8.0}
            }
        else:
            self.max_image_size = (1024, 1024)
            self.default_image_size = (768, 768)
            self.quality_settings = {
                'fast': {'steps': 30, 'guidance': 7.5},
                'balanced': {'steps': 50, 'guidance': 8.5},
                'high': {'steps': 75, 'guidance': 9.5}
            }
        self.style_prompt = (
            "professional concept art, architectural illustration, detailed textures, "
            "warm lighting, decorative elements, elegant design, artistic composition, "
            "Giramille signature style, vibrant colors, ornate details"
        )
        self.negative_prompt = (
            "low quality, blurry, distorted, bad anatomy, watermark, simple, basic, "
            "flat colors, missing details, poor lighting, amateur"
        )
        self.model_id = (
            "runwayml/stable-diffusion-v1-5" if device_type == "cpu"
            else "stabilityai/stable-diffusion-xl-base-1.0"
        )
        self.redis_url = os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0')
        self.model_cache_dir = 'models/cache'
        self.output_dir = 'outputs/production'

class ImageCache:
    """Redis-based image cache for production"""
    def __init__(self, redis_url: str):
        self.redis_client = None
        self.memory_cache = {}
        try:
            for i in range(3):
                try:
                    self.redis_client = redis.from_url(redis_url, socket_timeout=2.0)
                    self.redis_client.ping()
                    logger.info("[SUCCESS] Redis cache connected")
                    break
                except redis.ConnectionError as e:
                    if i == 2:
                        raise
                    logger.warning(f"Redis connection attempt {i+1} failed, retrying...")
                    time.sleep(1)
        except Exception as e:
            logger.warning(f"Redis not available, using memory cache: {e}")
            self.redis_client = None
    def get(self, key: str) -> Optional[bytes]:
        try:
            if self.redis_client:
                return self.redis_client.get(key)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    def set(self, key: str, value: bytes, ttl: int = 3600):
        try:
            if self.redis_client:
                self.redis_client.setex(key, ttl, value)
            else:
                self.memory_cache[key] = value
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    def generate_key(self, prompt: str, style: str, quality: str) -> str:
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
            total_success = self.metrics['requests_success']
            if total_success > 0:
                current_avg = self.metrics['avg_generation_time']
                self.metrics['avg_generation_time'] = (current_avg * (total_success - 1) + generation_time) / total_success
    def get_metrics(self) -> Dict:
        with self.lock:
            return self.metrics.copy()
    def get_system_info(self) -> Dict:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }

def _is_cuda_device(dev):
    import torch as _torch
    if isinstance(dev, str):
        return "cuda" in dev
    if isinstance(dev, _torch.device):
        return dev.type == "cuda"
    return False

class ProductionGiramilleGenerator:
    """Production-ready Giramille style generator"""
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.device = config.device_type
        self.cache = ImageCache(config.redis_url)
        self.monitor = PerformanceMonitor()
        self.request_queue = Queue(maxsize=config.max_concurrent_requests)
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        self.pipe = self._initialize_pipeline()
        self._setup_directories()
    def _initialize_pipeline(self):
        try:
            logger.info(f"Loading model {self.config.model_id} on {self.device}")
            trained_model_path = os.path.join(self.config.model_cache_dir, "giramille_style_model")
            if os.path.exists(trained_model_path):
                logger.info("Loading trained Giramille model...")
                model_id = trained_model_path
            else:
                logger.info("Loading base model...")
                model_id = self.config.model_id
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None
            )
            pipe = pipe.to(self.device)
            pipe.enable_attention_slicing(slice_size="auto")
            if self.device == "cuda" and hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                # try to enable xformers memory efficient attention, but continue if unavailable
                try:
                    import importlib
                    if importlib.util.find_spec("xformers") is not None:
                        try:
                            pipe.enable_xformers_memory_efficient_attention()
                            logger.info("Enabled xformers memory-efficient attention")
                        except Exception as e:
                            logger.warning(f"xformers found but enable failed, continuing without it: {e}")
                    else:
                        logger.info("xformers not installed; running without memory-efficient attention")
                except Exception as e:
                    logger.warning(f"Skipping xformers enablement due to error: {e}")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            logger.info("Model loaded successfully")
            return pipe
        except Exception as e:
            logger.error(f"Failed to initialize model pipeline: {e}")
            raise
    def _setup_directories(self):
        os.makedirs(self.config.model_cache_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    def _create_generator(self, seed: int):
        # create a torch Generator with a device that exists and is supported
        try:
            if _is_cuda_device(self.device):
                return torch.Generator(device="cuda").manual_seed(int(seed))
            else:
                return torch.Generator(device="cpu").manual_seed(int(seed))
        except Exception:
            # fallback to CPU generator or None if generator API not supported
            try:
                return torch.Generator().manual_seed(int(seed))
            except Exception:
                return None

    def generate_image(self, prompt: str, *, negative_prompt: Optional[str] = None, seed: Optional[int] = None, style: Optional[str] = None, quality: Optional[str] = None):
        """Generate an image and return a dict; robust to None inputs."""
        try:
            if prompt is None:
                raise ValueError("prompt must be provided")

            seed = int(seed) if seed is not None else 42
            gen = self._create_generator(seed)

            # normalize optional strings
            neg = negative_prompt or ""
            st = (style or "").lower()
            q = (quality or "").lower()

            # sampling params (can be adapted from quality)
            steps = 50 if q != "high" else 100
            guidance = 8.5 if q != "high" else 9.5

            # call pipeline; pass generator only if not None
            kwargs = dict(num_inference_steps=steps, guidance_scale=guidance)
            if gen is not None:
                kwargs["generator"] = gen
            if neg:
                kwargs["negative_prompt"] = neg

            out = self.pipe(prompt, **kwargs)
            img = out.images[0] if hasattr(out, "images") else out

            # convert PIL image to bytes for consistent return type
            img_bytes = None
            if isinstance(img, Image.Image):
                buf = BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            elif isinstance(img, (bytes, bytearray)):
                img_bytes = bytes(img)
            else:
                # try to convert numeric arrays
                try:
                    pil = Image.fromarray(img)
                    buf = BytesIO()
                    pil.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                except Exception:
                    img_bytes = None

            if img_bytes is None:
                raise RuntimeError("failed to obtain image bytes from pipeline output")

            return {
                "success": True,
                "image": img_bytes,
                "generation_time": None,
                "prompt": prompt,
                "style": style,
                "quality": quality,
            }
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {"success": False, "error": str(e), "prompt": prompt, "style": style, "quality": quality}
    def get_health_status(self) -> Dict:
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
        return self.monitor.get_metrics()
    def clear_cache(self):
        try:
            if self.cache.redis_client:
                self.cache.redis_client.flushdb()
                logger.info("[SUCCESS] Cache cleared")
            else:
                self.cache.memory_cache.clear()
                logger.info("[SUCCESS] Memory cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
