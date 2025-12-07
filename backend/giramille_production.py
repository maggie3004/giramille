__all__ = ['ProductionConfig', 'initialize_production_system']
"""
Production-Ready Giramille Style System
Optimized for production deployment with caching, monitoring, and error handling
"""

import warnings
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw
import numpy as np
import os
import time
import logging
from typing import Optional, Any, Dict
from io import BytesIO

import torch
from PIL import Image, ImageFilter, ImageEnhance

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logger = logging.getLogger("giramille.production")

# Defaults - change MODEL_ID to your local model path if you use local weights
MODEL_ID = os.environ.get("GIRAMILLE_MODEL_DIR", "runwayml/stable-diffusion-v1-5")
# optional upscaler: install realesrgan package & model if you want neural upscaling
TRY_REAL_ESRGAN = True

class ProductionGenerator:
    def __init__(self, pipe: StableDiffusionPipeline, device: torch.device, use_fp16: bool = False):
        self.pipe = pipe
        self.device = device
        self.use_fp16 = use_fp16

        # prefer a stable scheduler for high quality
        try:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            pass

        # warm-up small call
        logger.info("ProductionGenerator initialized on %s (fp16=%s)", device, use_fp16)

    def _postprocess_image(self, img: Image.Image, quality: str):
        if quality == "high":
            # upscale 2x with Lanczos + sharpen/contrast
            w, h = img.size
            img = img.resize((w * 2, h * 2), resample=Image.LANCZOS)
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            img = ImageEnhance.Contrast(img).enhance(1.05)
            img = ImageEnhance.Color(img).enhance(1.05)

            # optional: call Real-ESRGAN if installed (best results with GPU)
            if TRY_REAL_ESRGAN:
                try:
                    from realesrgan import RealESRGAN
                    device = "cuda" if self.device.type == "cuda" else "cpu"
                    with RealESRGAN(device, scale=2) as rr:
                        rr.load_weights("RealESRGAN_x2plus")  # ensure model weights installed
                        arr = rr.predict(img)
                        img = Image.fromarray(arr)
                except Exception as e:
                    logger.debug("Real-ESRGAN unavailable or failed: %s", e)
        return img

    def generate_image(self, prompt: str, *, negative_prompt: Optional[str] = None,
                       seed: Optional[int] = None, style: Optional[str] = None,
                       quality: Optional[str] = "high", width: int = 512, height: int = 512) -> Dict[str, Any]:
        try:
            seed = int(seed) if seed is not None else int(time.time()) & 0xFFFFFFFF
            generator = torch.Generator(self.device).manual_seed(seed) if self.device.type == "cuda" or self.device.type == "cpu" else None

            q = (quality or "balanced").lower()
            if q == "high":
                steps = 60
                guidance = 12.0
            elif q == "balanced":
                steps = 40
                guidance = 9.0
            else:
                steps = 25
                guidance = 7.5

            logger.info("generate_image: seed=%s steps=%s guidance=%s quality=%s", seed, steps, guidance, q)
            pipe_call = self.pipe.to(self.device)

            call_kwargs = dict(prompt=prompt,
                               negative_prompt=negative_prompt,
                               height=height,
                               width=width,
                               num_inference_steps=int(steps),
                               guidance_scale=float(guidance),
                               generator=generator)

            # run with fp16/autocast on CUDA for speed/quality
            if self.use_fp16 and self.device.type == "cuda":
                with torch.autocast("cuda"):
                    out = pipe_call(**call_kwargs)
            else:
                out = pipe_call(**call_kwargs)

            image = out.images[0] if hasattr(out, "images") else out
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # postprocess/upscale for high quality
            image = self._postprocess_image(image, q)

            buf = BytesIO()
            image.save(buf, format="PNG", optimize=True)
            img_bytes = buf.getvalue()

            return {"success": True, "image": img_bytes, "final_prompt": prompt, "seed": seed, "quality": q}
        except Exception as e:
            logger.exception("generate_image failed")
            return {"success": False, "error": str(e)}

def initialize_production_system(device: Optional[str] = None) -> ProductionGenerator:
    """
    Load pipeline and return a ProductionGenerator.
    device: "cuda"|"cpu"|None
    """
    # detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    use_fp16 = True if device.type == "cuda" else False

    logger.info("Loading model '%s' on %s (fp16=%s)", MODEL_ID, device, use_fp16)
    # load pipeline: prefer local dir if MODEL_ID points to local weights
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=(torch.float16 if use_fp16 else torch.float32))
    # move to device
    pipe = pipe.to(device)

    # reduce safety checker behavior only if you explicitly want it disabled
    # pipe.safety_checker = None  # consider enabling in production with moderation

    gen = ProductionGenerator(pipe, device, use_fp16)
    return gen

__all__ = ['initialize_production_system', 'ProductionGenerator']
