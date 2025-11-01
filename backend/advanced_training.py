"""
Advanced Training System for Giramille Style
This will train the model specifically for Giramille style with better color accuracy
"""
# core imports
import logging
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

# third-party
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GiramilleStyleDataset(Dataset):
    """
    Minimal dataset placeholder. Replace with real dataset logic that yields (prompt, image_path) pairs
    or only prompts depending on your training strategy (fine-tune vs. evaluation).
    """

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class GiramilleStyleValidator:
    """Validator for Giramille style quality (CLIP + simple color metric)."""

    def __init__(self, device: Optional[torch.device | str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # load CLIP for text-image scoring; failures degrade gracefully
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            logger.warning(f"Failed to load CLIP for validation: {e}")
            self.clip_model = None
            self.clip_processor = None

    def validate_clip_similarity(self, prompt: str, image: Image.Image) -> float:
        """Return CLIP cosine similarity normalized to 0..1 (higher = text matches image)."""
        if self.clip_model is None or self.clip_processor is None:
            return 0.0
        try:
            inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            text_emb = outputs.text_embeds
            image_emb = outputs.image_embeds
            sim = torch.nn.functional.cosine_similarity(text_emb, image_emb).cpu().item()
            return float((sim + 1.0) / 2.0)
        except Exception as e:
            logger.debug(f"CLIP similarity error: {e}")
            return 0.0

    def validate_color_accuracy(self, image: Image.Image) -> float:
        """
        Simple colorfulness proxy: mean saturation-weighted value.
        Returns value in 0..1.
        """
        try:
            im = image.convert("RGB")
            arr = np.asarray(im).astype(np.float32) / 255.0
            maxc = arr.max(axis=2)
            minc = arr.min(axis=2)
            saturation = np.where(maxc == 0, 0, (maxc - minc) / (maxc + 1e-8))
            value = maxc
            score = float(np.clip(saturation.mean() * 0.7 + value.mean() * 0.3, 0.0, 1.0))
            return score
        except Exception as e:
            logger.debug(f"Color accuracy error: {e}")
            return 0.0


def generate_best_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    out_dir: str,
    epoch: int,
    batch_idx: int,
    device: Optional[torch.device | str] = None,
    seeds: Optional[List[int]] = None,
    num_inference_steps: int = 100,
    guidance_scale: float = 10,
    height: int = 512,
    width: int = 512,
) -> Tuple[float, Optional[Image.Image]]:
    """
    Generate multiple samples for a prompt, score them with CLIP + color metric,
    and save the best image to disk.
    Returns (best_score, best_image).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    validator = GiramilleStyleValidator(device=device)
    gen_params = dict(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width)

    best_score = -1.0
    best_image = None
    seeds = seeds or [42, random.randint(0, 2**31 - 1), random.randint(0, 2**31 - 1)]

    for seed in seeds:
        try:
            # create generator when pipeline accepts it; generator device must match pipeline device
            gen = None
            try:
                gen = torch.Generator(device="cuda" if "cuda" in str(device) else "cpu").manual_seed(int(seed))
            except Exception:
                gen = None
            with torch.no_grad():
                result = pipe(prompt, generator=gen, **gen_params)
            image = result.images[0]
            clip_score = validator.validate_clip_similarity(prompt, image)
            color_score = validator.validate_color_accuracy(image)
            combined = 0.7 * clip_score + 0.3 * color_score
            logger.debug(f"seed={seed} clip={clip_score:.3f} color={color_score:.3f} combined={combined:.3f}")
            if combined > best_score:
                best_score = combined
                best_image = image
        except Exception as e:
            logger.warning(f"generation error (seed {seed}): {e}")

    if best_image is not None:
        os.makedirs(out_dir, exist_ok=True)
        safe_prompt = "".join(c if c.isalnum() or c in "-_." else "_" for c in prompt)[:120]
        filename = f"epoch_{epoch}_batch_{batch_idx}_{safe_prompt}_score_{best_score:.3f}.png"
        path = os.path.join(out_dir, filename)
        try:
            best_image.save(path)
            logger.info(f"Saved best image: {path} (score={best_score:.3f})")
        except Exception as e:
            logger.warning(f"Failed to save image {path}: {e}")
    else:
        logger.warning(f"No image generated for prompt: {prompt}")

    return best_score, best_image


class GiramilleTrainer:
    """
    Lightweight trainer wrapper that currently uses generation + validation
    to select best samples. This is NOT weight fine-tuning; use LoRA/PEFT or DreamBooth
    for actual training of model weights.
    """

    def __init__(self, pipe: StableDiffusionPipeline, device: Optional[torch.device | str] = None):
        self.pipe = pipe
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ensure pipeline device
        try:
            self.pipe = self.pipe.to(self.device)
        except Exception:
            logger.debug("Could not move pipeline to device; continuing with default.")

    def train(self, prompts: List[str], epochs: int = 4, batch_size: int = 1):
        ds = GiramilleStyleDataset(prompts)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs} start - {datetime.utcnow().isoformat()}Z")
            batch_idx = 0
            for batch in dl:
                # batch may be list of prompts
                for prompt in (batch if isinstance(batch, (list, tuple)) else [batch]):
                    score = self._train_single_prompt(prompt, epoch, batch_idx)
                    logger.info(f"Epoch {epoch} batch {batch_idx} prompt score: {score:.3f}")
                    batch_idx += 1

    def _train_single_prompt(self, prompt: str, epoch: int, batch_idx: int) -> float:
        out_dir = os.path.join("outputs", "training", f"epoch_{epoch}")
        score, _ = generate_best_image(self.pipe, prompt, out_dir, epoch, batch_idx, device=self.device)
        return float(score or 0.0)


if __name__ == "__main__":
    # minimal runnable example: only run if a local pipeline is available.
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # attempt to load a small pipeline if user configured; replace the model id as needed
        model_id = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
        logger.info(f"Loading pipeline {model_id} on {device} (this may download weights)...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        # set scheduler to DPMSolverMultistep for slightly better sampling (optional)
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            logger.debug("Could not switch scheduler; using pipeline default.")
        trainer = GiramilleTrainer(pipe, device=device)
        sample_prompts = [
            "A colorful sunset over the ocean in Giramille style, vivid saturation",
            "Portrait in Giramille style, simplified shapes and high contrast colors",
        ]
        trainer.train(sample_prompts, epochs=1, batch_size=1)
    except Exception as e:
        logger.error(f"Runner failed: {e}")
