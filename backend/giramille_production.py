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

        # enable memory-efficient attention / xformers if available
        try:
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.debug("xformers not enabled or unavailable")

        # enable attention slicing to reduce peak memory (helps on modest GPUs)
        try:
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
        except Exception:
            pass

        # ensure pipeline is on target device (should be moved in initializer)
        try:
            self.pipe = self.pipe.to(self.device)
        except Exception:
            logger.debug("Failed to move pipeline to device during init")

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
                    if self.device.type == "cuda":
                        rr = RealESRGAN("cuda", scale=2)
                    else:
                        rr = RealESRGAN("cpu", scale=2)
                    try:
                        rr.load_weights("RealESRGAN_x2plus")  # ensure model weights installed
                        arr = rr.predict(img)
                        img = Image.fromarray(arr)
                    finally:
                        try:
                            rr.close()
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug("Real-ESRGAN unavailable or failed: %s", e)
        return img

    def generate_image(self, prompt: str, *, negative_prompt: Optional[str] = None,
                       seed: Optional[int] = None, style: Optional[str] = None,
                       quality: Optional[str] = "high", width: int = 512, height: int = 512) -> Dict[str, Any]:
        try:
            seed = int(seed) if seed is not None else int(time.time()) & 0xFFFFFFFF
            # prepare generator for deterministic outputs when supported
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except Exception:
                # fallback for older torch APIs
                generator = torch.Generator().manual_seed(seed)

            q = (quality or "balanced").lower()
            if q == "high":
                steps = 60
                guidance = 10.0
                # prefer larger base resolution when GPU available
                if self.device.type == "cuda" and width < 768:
                    width = 768
                    height = 768
            elif q == "balanced":
                steps = 40
                guidance = 9.0
            else:
                steps = 25
                guidance = 7.5

            logger.info("generate_image: seed=%s steps=%s guidance=%s quality=%s device=%s", seed, steps, guidance, q, self.device)

            call_kwargs = dict(prompt=prompt,
                               negative_prompt=negative_prompt,
                               height=height,
                               width=width,
                               num_inference_steps=int(steps),
                               guidance_scale=float(guidance),
                               generator=generator)

            # use the initialized pipeline on the correct device
            pipe_call = self.pipe

            # run with fp16/autocast on CUDA for speed/quality
            if self.use_fp16 and self.device.type == "cuda":
                with torch.autocast(self.device.type):
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
    # Before loading, perform a quick check for common safetensors/bin files when MODEL_ID is a local path
    def _check_model_files(model_id: str) -> Dict[str, bool]:
        # returns mapping component -> exists(bool)
        components = {
            "unet": ["unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.bin"],
            "text_encoder": ["text_encoder/model.safetensors", "text_encoder/pytorch_model.bin"],
            "vae": ["vae/vae.safetensors", "vae/diffusion_pytorch_model.safetensors", "vae/pytorch_model.bin"]
        }
        results = {}
        # Only check when model_id appears to be a local path
        if os.path.exists(model_id):
            for comp, candidates in components.items():
                found = False
                for c in candidates:
                    if os.path.exists(os.path.join(model_id, c)):
                        found = True
                        break
                results[comp] = found
        else:
            # remote repo; don't assume local files
            for comp in components.keys():
                results[comp] = False
        return results

    try:
        file_check = _check_model_files(MODEL_ID)
        missing = [k for k, v in file_check.items() if not v]
        if missing and os.path.exists(MODEL_ID):
            logger.warning("Model directory '%s' is missing component files for: %s. Diffusers may attempt remote download.", MODEL_ID, ",".join(missing))
    except Exception:
        logger.debug("Model file check failed, continuing to load pipeline")

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=(torch.float16 if use_fp16 else torch.float32))
    # move to device
    pipe = pipe.to(device)

    # reduce safety checker behavior only if you explicitly want it disabled
    # pipe.safety_checker = None  # consider enabling in production with moderation

    gen = ProductionGenerator(pipe, device, use_fp16)
    return gen

__all__ = ['initialize_production_system', 'ProductionGenerator']
