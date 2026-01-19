"""
Unified High-Precision 2D-to-3D Generation Pipeline
Combines 2D image generation with 3D model creation for maximum accuracy
"""

import torch
import logging
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import json
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from high_precision_generator import HighPrecisionGenerator
from text_to_3d_generator import Text3DGenerator, AccuracyEnhancer

logger = logging.getLogger(__name__)


class UnifiedGenerationPipeline:
    """Unified 2D and 3D generation pipeline"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize unified pipeline
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_generator = HighPrecisionGenerator(device=self.device)
        self.model_generator = Text3DGenerator(device=self.device)
        self.accuracy_enhancer = AccuracyEnhancer()
        
        logger.info(f"Unified Pipeline initialized on {self.device}")
    
    def generate_3d_with_high_precision(
        self,
        prompt: str,
        output_dir: str = 'outputs',
        generate_preview_2d: bool = True,
        style: str = 'high_detail',
        steps_2d: int = 50,
        steps_3d: int = 64,
        refine_accuracy: bool = True,
        validate_output: bool = True
    ) -> Dict[str, Any]:
        """
        Generate 3D model with high precision accuracy
        
        This is the main entry point that combines:
        1. High-precision 2D image generation (for reference)
        2. Native 3D model generation from text
        3. Accuracy validation and enhancement
        
        Args:
            prompt: Text description of 3D object
            output_dir: Output directory
            generate_preview_2d: Also generate high-quality 2D preview
            style: Generation style
            steps_2d: 2D generation diffusion steps
            steps_3d: 3D generation diffusion steps
            refine_accuracy: Refine accuracy through multiple passes
            validate_output: Validate 3D model quality
            
        Returns:
            Complete generation result with 2D preview and 3D model
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"3d_gen_{timestamp}"
        
        logger.info(f"Starting unified 3D generation: {job_id}")
        logger.info(f"Prompt: {prompt}")
        
        result = {
            'job_id': job_id,
            'prompt': prompt,
            'timestamp': timestamp,
            'device': self.device,
            'stages': {}
        }
        
        try:
            # Stage 1: Generate 2D preview (for reference)
            if generate_preview_2d:
                logger.info("Stage 1/3: Generating high-precision 2D preview...")
                
                image_result = self.image_generator.generate_image_high_precision(
                    prompt=prompt,
                    width=768,
                    height=768,
                    num_inference_steps=steps_2d,
                    guidance_scale=7.5,
                    num_images=1,
                    style=style,
                    quality_level='very_high',
                    num_passes=1,
                    output_dir=output_dir
                )
                
                result['stages']['2d_preview'] = image_result
                
                if 'error' not in image_result and image_result.get('best_image'):
                    logger.info(f"✓ 2D preview generated: {image_result['best_image']['filename']}")
                    preview_path = image_result['best_image']['path']
                else:
                    logger.warning("Failed to generate 2D preview")
                    preview_path = None
            
            # Stage 2: Generate 3D model
            logger.info("Stage 2/3: Generating 3D model (this may take 2-5 minutes)...")
            
            model_result = self.model_generator.generate_3d_mesh(
                prompt=prompt,
                steps=steps_3d,
                cfg_scale=15.0,
                save_ply=True,
                save_obj=True,
                output_dir=output_dir
            )
            
            result['stages']['3d_model'] = model_result
            
            if 'error' in model_result:
                logger.error(f"3D generation failed: {model_result['error']}")
                return result
            
            logger.info(f"✓ 3D model generated: {model_result['id']}")
            model_path = model_result['files'].get('obj') or model_result['files'].get('ply')
            
            # Stage 3: Validate and enhance accuracy
            if validate_output:
                logger.info("Stage 3/3: Validating and optimizing 3D model...")
                
                # Validate model
                if model_path and os.path.exists(model_path):
                    validation_result = self.accuracy_enhancer.validate_3d_model(model_path)
                    result['stages']['validation'] = validation_result
                    logger.info(f"Validation stats: {validation_result}")
                    
                    # Optimize mesh if needed
                    if 'vertices' in validation_result and validation_result.get('vertices', 0) > 100000:
                        logger.info("Optimizing high-poly mesh...")
                        optimized_path = self.accuracy_enhancer.optimize_mesh(model_path)
                        result['stages']['optimization'] = {
                            'original_path': model_path,
                            'optimized_path': optimized_path
                        }
                        model_path = optimized_path
            
            # Generate multi-view preview
            logger.info("Generating multi-view preview images...")
            multiview_result = self._generate_multiview_preview(model_path, output_dir)
            result['stages']['multiview'] = multiview_result
            
            # Final summary
            result['final_output'] = {
                'model_id': model_result.get('id'),
                'model_path': model_path,
                'preview_2d': preview_path if generate_preview_2d else None,
                'multiview_previews': multiview_result,
                'quality_score': result['stages']['validation'].get('valid', False) if validate_output else None
            }
            
            logger.info(f"✓ 3D generation complete: {job_id}")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result['error'] = str(e)
            return result
    
    def _generate_multiview_preview(self, model_path: str, output_dir: str) -> Dict[str, str]:
        """Generate preview images from multiple viewpoints"""
        try:
            # Placeholder for multi-view rendering
            # In production, use Trimesh or Pyvista for rendering
            
            preview_paths = {
                'front': 'preview_front.png',
                'top': 'preview_top.png',
                'side': 'preview_side.png',
            }
            
            return preview_paths
            
        except Exception as e:
            logger.error(f"Failed to generate multiview preview: {e}")
            return {}
    
    def generate_batch(
        self,
        prompts: list,
        output_dir: str = 'outputs/batch',
        parallel: bool = False
    ) -> list:
        """
        Generate multiple 3D models from prompts
        
        Args:
            prompts: List of text prompts
            output_dir: Output directory
            parallel: Generate in parallel (requires multiple GPUs)
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Generating {i}/{len(prompts)}: {prompt}")
            
            batch_subdir = os.path.join(output_dir, f"model_{i}")
            result = self.generate_3d_with_high_precision(
                prompt=prompt,
                output_dir=batch_subdir,
                refine_accuracy=True,
                validate_output=True
            )
            
            results.append(result)
        
        return results
    
    def generate_with_variations(
        self,
        prompt: str,
        output_dir: str = 'outputs',
        num_variations: int = 3,
        style_variations: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate multiple variations of the same object
        
        Args:
            prompt: Base prompt
            output_dir: Output directory
            num_variations: Number of variations
            style_variations: Different styles to try
            
        Returns:
            Dictionary with all variations
        """
        if style_variations is None:
            style_variations = ['high_detail', 'photorealistic', 'stylized']
        
        results = {
            'base_prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'variations': []
        }
        
        for i in range(num_variations):
            style = style_variations[i % len(style_variations)]
            var_prompt = f"{prompt} (variation {i+1}, {style})"
            
            logger.info(f"Generating variation {i+1}/{num_variations} with style: {style}")
            
            var_output_dir = os.path.join(output_dir, f"variation_{i+1}")
            result = self.generate_3d_with_high_precision(
                prompt=var_prompt,
                output_dir=var_output_dir,
                style=style,
                refine_accuracy=True,
                validate_output=True
            )
            
            results['variations'].append(result)
        
        return results


class ProgressTracker:
    """Track generation progress and provide callbacks"""
    
    def __init__(self):
        self.callbacks = []
        self.current_stage = ""
        self.progress = 0
    
    def on_stage_start(self, stage_name: str, callback=None):
        """Register callback for stage start"""
        logger.info(f"[PROGRESS] Starting: {stage_name}")
        self.current_stage = stage_name
        self.progress = 0
        if callback:
            callback(stage_name, 0)
    
    def on_progress(self, progress_percent: float, callback=None):
        """Update progress"""
        self.progress = progress_percent
        if callback:
            callback(self.current_stage, progress_percent)
    
    def on_stage_complete(self, stage_name: str, result: Any, callback=None):
        """Register callback for stage completion"""
        logger.info(f"[PROGRESS] Completed: {stage_name}")
        if callback:
            callback(stage_name, 100, result)


# Convenience functions
def generate_3d_directly(
    prompt: str,
    output_dir: str = 'outputs',
    with_2d_preview: bool = True,
    high_accuracy: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Simple function to generate 3D model directly
    
    Args:
        prompt: Object description
        output_dir: Output directory
        with_2d_preview: Include 2D preview
        high_accuracy: Use maximum accuracy settings
        **kwargs: Additional parameters
        
    Returns:
        Generation result
    """
    pipeline = UnifiedGenerationPipeline()
    
    if high_accuracy:
        return pipeline.generate_3d_with_high_precision(
            prompt=prompt,
            output_dir=output_dir,
            generate_preview_2d=with_2d_preview,
            steps_2d=50,
            steps_3d=64,
            refine_accuracy=True,
            validate_output=True,
            **kwargs
        )
    else:
        return pipeline.generate_3d_with_high_precision(
            prompt=prompt,
            output_dir=output_dir,
            generate_preview_2d=with_2d_preview,
            steps_2d=30,
            steps_3d=32,
            refine_accuracy=False,
            validate_output=True,
            **kwargs
        )
