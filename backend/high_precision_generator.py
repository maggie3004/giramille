"""
High-Precision 2D Image Generation with Accuracy Optimization
Generates high-quality, accurate images with multiple quality assurance passes
"""

import torch
import numpy as np
import logging
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image
import json
import os
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from diffusers import (
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline
    )
    from transformers import CLIPProcessor, CLIPModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers not installed. Install with: pip install diffusers transformers")

try:
    from PIL import ImageEnhance, ImageFilter
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


@dataclass
class ImageQualityScore:
    """Quality assessment for generated images"""
    overall_score: float
    sharpness: float
    color_vibrancy: float
    object_clarity: float
    composition: float
    details: float
    issues: List[str]


class AdvancedPromptOptimizer:
    """Advanced prompt optimization for maximum accuracy"""
    
    def __init__(self):
        self.prompt_templates = {
            'base': "{description}",
            'high_detail': "{description}, highly detailed, intricate details, professional quality, 4k, 8k",
            'photorealistic': "{description}, photorealistic, ultra detailed, sharp focus, professional lighting, masterpiece",
            'stylized': "{description}, stylized illustration, professional art, vibrant colors, detailed, high quality",
            'cartoon': "{description}, cartoon style, bright colors, clean lines, professional illustration, high quality",
            'anime': "{description}, anime style, detailed, high quality, beautiful, vibrant colors",
            'concept_art': "{description}, concept art, highly detailed, professional, dramatic lighting, masterpiece"
        }
        
        self.detail_boosters = [
            "sharp focus",
            "intricate details",
            "fine detail",
            "high definition",
            "highly detailed",
            "masterpiece",
            "professional quality",
            "cinema lighting",
            "dramatic lighting",
            "volumetric lighting"
        ]
        
        self.quality_modifiers = {
            'very_low': "low quality, blurry, pixelated",
            'low': "average quality",
            'medium': "good quality, clear",
            'high': "high quality, sharp, clear, detailed",
            'very_high': "ultra high quality, 8k, masterpiece, professional, photorealistic"
        }
        
        self.negative_defaults = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "bad hands, three hands, three legs, bad arms, missing limbs, "
            "missing fingers, too many fingers, fused fingers, long body, "
            "duplicated body parts, gross proportions, disfigured, "
            "mutation, mutations, low res, bad quality, normal quality, "
            "worst quality, jpeg artifacts, signature, watermark, text"
        )
    
    def optimize_prompt(
        self,
        prompt: str,
        style: str = 'high_detail',
        quality_level: str = 'very_high',
        add_keywords: Optional[List[str]] = None
    ) -> str:
        """
        Optimize prompt for maximum generation accuracy
        
        Args:
            prompt: User's original prompt
            style: Style template to use
            quality_level: Quality level (very_low, low, medium, high, very_high)
            add_keywords: Additional keywords to add
            
        Returns:
            Optimized prompt
        """
        # Clean up prompt
        prompt = prompt.strip()
        if not prompt:
            prompt = "a beautiful landscape"
        
        # Apply template
        template = self.prompt_templates.get(style, self.prompt_templates['high_detail'])
        optimized = template.format(description=prompt)
        
        # Add quality modifier
        if quality_level in self.quality_modifiers:
            optimized += f", {self.quality_modifiers[quality_level]}"
        
        # Add style-specific boosters
        if style == 'photorealistic':
            optimized += ", volumetric lighting, cinematic, professional photography"
        elif style == 'anime':
            optimized += ", anime aesthetic, beautiful, clean art style"
        elif style == 'cartoon':
            optimized += ", cartoon aesthetic, playful, colorful"
        elif style == 'concept_art':
            optimized += ", concept art style, professional, highly detailed"
        
        # Add user-specified keywords
        if add_keywords:
            optimized += ", " + ", ".join(add_keywords)
        
        # Remove duplicates
        words = optimized.split(", ")
        words = list(dict.fromkeys(words))  # Remove duplicates while preserving order
        optimized = ", ".join(words)
        
        return optimized
    
    def get_negative_prompt(self, add_negative: Optional[str] = None) -> str:
        """Get negative prompt for improved quality"""
        negative = self.negative_defaults
        if add_negative:
            negative += f", {add_negative}"
        return negative


class HighPrecisionGenerator:
    """High-precision 2D image generation engine"""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
        """
        Initialize high-precision generator
        
        Args:
            model_id: HuggingFace model ID
            device: 'cuda' or 'cpu'
        """
        self.model_id = model_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_optimizer = AdvancedPromptOptimizer()
        
        self.pipe = None
        self.quality_assessor = None
        self.initialized = False
        
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load diffusion pipeline"""
        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers not available")
            return
        
        try:
            logger.info(f"Loading Stable Diffusion from {self.model_id}...")
            
            # Load pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None  # Disable safety checker for faster generation
            )
            
            # Use better scheduler for quality
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            if self.device == 'cuda':
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    logger.debug("xformers not available")
            
            self.initialized = True
            logger.info(f"✓ Pipeline loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            self.initialized = False
    
    def generate_image_high_precision(
        self,
        prompt: str,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        style: str = 'high_detail',
        quality_level: str = 'very_high',
        return_quality_score: bool = True,
        num_passes: int = 1,
        output_dir: str = 'outputs'
    ) -> Dict[str, Any]:
        """
        Generate high-precision images with quality assurance
        
        Args:
            prompt: Text description
            width: Image width (768 for high quality)
            height: Image height
            num_inference_steps: Diffusion steps (50+ for quality)
            guidance_scale: Classifier-free guidance scale
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            style: Generation style
            quality_level: Quality level
            return_quality_score: Return quality assessment
            num_passes: Number of generation passes to select best
            output_dir: Output directory
            
        Returns:
            Dictionary with images and quality metrics
        """
        if not self.initialized:
            return {'error': 'Pipeline not loaded'}
        
        try:
            # Optimize prompt
            optimized_prompt = self.prompt_optimizer.optimize_prompt(
                prompt,
                style=style,
                quality_level=quality_level
            )
            negative_prompt = self.prompt_optimizer.get_negative_prompt()
            
            logger.info(f"Original prompt: {prompt}")
            logger.info(f"Optimized prompt: {optimized_prompt}")
            
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id = f"image_{timestamp}"
            
            best_image = None
            best_score = -1
            all_results = []
            
            # Generate multiple passes and select best
            for pass_num in range(num_passes):
                logger.info(f"Generation pass {pass_num + 1}/{num_passes}...")
                
                # Set seed for reproducibility
                if seed is not None:
                    torch.manual_seed(seed + pass_num)
                    np.random.seed(seed + pass_num)
                
                # Generate
                with torch.no_grad():
                    result = self.pipe(
                        prompt=optimized_prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images,
                        output_type='pil'
                    )
                
                images = result.images
                
                # Assess quality and select best
                for img_idx, image in enumerate(images):
                    # Post-process for quality
                    image = self._post_process_image(image, quality_level)
                    
                    # Assess quality
                    if return_quality_score:
                        score = self._assess_image_quality(image)
                        quality_text = f"Score: {score.overall_score:.2f}"
                    else:
                        score = None
                        quality_text = ""
                    
                    # Save image
                    filename = f"{batch_id}_pass{pass_num}_img{img_idx}.png"
                    filepath = os.path.join(output_dir, filename)
                    image.save(filepath, quality=95)
                    
                    result_entry = {
                        'path': filepath,
                        'filename': filename,
                        'prompt': prompt,
                        'optimized_prompt': optimized_prompt,
                        'pass': pass_num,
                        'index': img_idx,
                        'size': (width, height),
                        'quality_score': score.overall_score if score else None,
                        'quality_details': score.__dict__ if score else None
                    }
                    all_results.append(result_entry)
                    
                    # Track best image
                    if score and score.overall_score > best_score:
                        best_score = score.overall_score
                        best_image = result_entry
                    
                    logger.info(f"✓ Generated {filename} {quality_text}")
            
            # Save batch metadata
            batch_metadata = {
                'batch_id': batch_id,
                'prompt': prompt,
                'optimized_prompt': optimized_prompt,
                'timestamp': timestamp,
                'style': style,
                'quality_level': quality_level,
                'device': self.device,
                'parameters': {
                    'width': width,
                    'height': height,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale,
                    'num_images': num_images,
                    'seed': seed
                },
                'best_image': best_image,
                'all_results': all_results
            }
            
            meta_path = os.path.join(output_dir, f"{batch_id}_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(batch_metadata, f, indent=2, default=str)
            
            batch_metadata['metadata_file'] = meta_path
            
            logger.info(f"✓ Batch complete: {len(all_results)} images generated")
            return batch_metadata
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {'error': str(e)}
    
    def _post_process_image(self, image: Image.Image, quality_level: str) -> Image.Image:
        """Post-process image for maximum quality"""
        if not PILLOW_AVAILABLE:
            return image
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)  # 30% more sharpness
        
        # Enhance color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.15)  # 15% more vibrant
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)  # 10% more contrast
        
        # Quality-specific processing
        if quality_level == 'very_high':
            # Apply unsharp mask for extra detail
            image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=200, threshold=3))
        elif quality_level == 'high':
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=2))
        
        return image
    
    def _assess_image_quality(self, image: Image.Image) -> ImageQualityScore:
        """Assess image quality using multiple metrics"""
        issues = []
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Sharpness assessment (using Laplacian variance)
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        laplacian = np.var(np.convolve(gray.flatten(), [1, -1], mode='same'))
        sharpness = min(laplacian / 100, 1.0)
        if sharpness < 0.3:
            issues.append("Low sharpness")
        
        # Color vibrancy
        if len(img_array.shape) == 3:
            color_std = np.std(img_array)
            color_vibrancy = min(color_std / 50, 1.0)
            if color_vibrancy < 0.3:
                issues.append("Low color vibrancy")
        else:
            color_vibrancy = 0.5
        
        # Object clarity (using edge detection)
        from scipy import ndimage
        edges = np.abs(ndimage.sobel(gray))
        object_clarity = np.mean(edges) / 255
        if object_clarity < 0.1:
            issues.append("Low object clarity")
        
        # Overall metrics
        composition = 0.7  # Default composition score
        details = min((laplacian / 50), 1.0)
        
        # Calculate overall score
        overall_score = (
            sharpness * 0.25 +
            color_vibrancy * 0.2 +
            object_clarity * 0.25 +
            composition * 0.15 +
            details * 0.15
        )
        
        return ImageQualityScore(
            overall_score=overall_score,
            sharpness=sharpness,
            color_vibrancy=color_vibrancy,
            object_clarity=object_clarity,
            composition=composition,
            details=details,
            issues=issues
        )


def generate_high_precision_image(
    prompt: str,
    style: str = 'high_detail',
    quality_level: str = 'very_high',
    output_dir: str = 'outputs',
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to generate high-precision image"""
    generator = HighPrecisionGenerator()
    return generator.generate_image_high_precision(
        prompt,
        style=style,
        quality_level=quality_level,
        output_dir=output_dir,
        **kwargs
    )
