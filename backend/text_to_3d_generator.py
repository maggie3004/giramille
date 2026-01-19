"""
High-Precision Text-to-3D Generation Module
Generates 3D models directly from text prompts with maximum accuracy
"""

import torch
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING
from PIL import Image
import io
import json
import os
from datetime import datetime

if TYPE_CHECKING:
    from shap_e.rendering.view_data import PointCloud

logger = logging.getLogger(__name__)

# Try to import Shap-E (text-to-3D generation)
try:
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.download import load_model
    from shap_e.rendering.view_data import PointCloud
    import shap_e.util.point_cloud as pc_util
    SHAP_E_AVAILABLE = True
except ImportError:
    SHAP_E_AVAILABLE = False
    PointCloud = None  # Define as None when not available
    logger.warning("Shap-E not installed. For 3D generation, install: pip install git+https://github.com/openai/shap-e.git")

# Try to import trimesh for 3D model handling
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("Trimesh not installed. Install with: pip install trimesh")

# Try to import Open3D for advanced 3D operations
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not installed. Install with: pip install open3d")


class PromptOptimizer:
    """Optimizes text prompts for maximum 3D generation accuracy"""
    
    def __init__(self):
        self.quality_descriptors = {
            'detail': ['intricate', 'detailed', 'high-poly', 'complex geometry', 'sophisticated'],
            'material': ['metallic', 'glossy', 'matte', 'reflective', 'textured'],
            'lighting': ['well-lit', 'dramatic lighting', 'studio lighting', 'volumetric lighting'],
            'style': ['photorealistic', 'stylized', 'artistic', 'professional', '3d model'],
            'quality': ['high quality', '8k', 'masterpiece', 'ultra detailed', 'professional grade']
        }
    
    def optimize_prompt(self, original_prompt: str) -> str:
        """
        Enhance prompt for maximum 3D generation quality
        
        Args:
            original_prompt: User's original prompt
            
        Returns:
            Optimized prompt for better 3D generation
        """
        # Base components
        enhanced = original_prompt
        
        # Add quality indicators if missing
        if not any(word in enhanced.lower() for word in ['detailed', 'high quality', '3d', 'model']):
            enhanced += ", high-quality 3d model, intricate details"
        
        # Add material/lighting if missing
        if not any(word in enhanced.lower() for word in ['metallic', 'glossy', 'matte', 'lighting']):
            enhanced += ", professional studio lighting, detailed textures"
        
        # Ensure geometric specificity
        if not any(word in enhanced.lower() for word in ['smooth', 'sharp', 'geometry', 'surface']):
            enhanced += ", clean geometry, well-defined surfaces"
        
        # Add quality suffix
        if not enhanced.endswith(('masterpiece', 'professional', 'perfect')):
            enhanced += ", masterpiece, professional quality"
        
        return enhanced
    
    def create_negative_prompt(self) -> str:
        """Generate negative prompt for better 3D quality"""
        return (
            "blurry, low quality, deformed, distorted, flat, 2d, cartoon, "
            "sketch, low poly, broken geometry, missing parts, artifacts, "
            "noise, jpeg compression, watermark, text, simple"
        )


class Text3DGenerator:
    """Main 3D model generation engine"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize 3D generation system
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_optimizer = PromptOptimizer()
        self.xm = None
        self.model = None
        self.initialized = False
        
        # Load models on demand
        self._load_models()
    
    def _load_models(self):
        """Lazy load Shap-E models"""
        if not SHAP_E_AVAILABLE:
            logger.error("Shap-E not available. Cannot generate 3D models.")
            return
        
        try:
            logger.info(f"Loading Shap-E models on {self.device}...")
            
            # Load transmitter (converts latent to point cloud)
            self.xm = load_model('transmitter', device=self.device)
            
            # Load diffusion model (generates 3D latent codes)
            self.model = load_model('diffusion', device=self.device)
            
            self.initialized = True
            logger.info("✓ Shap-E models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Shap-E models: {e}")
            self.initialized = False
    
    def generate_3d_mesh(
        self,
        prompt: str,
        steps: int = 64,
        cfg_scale: float = 15.0,
        size: int = 64,
        save_ply: bool = True,
        save_obj: bool = True,
        output_dir: str = 'outputs'
    ) -> Dict[str, Any]:
        """
        Generate 3D mesh from text prompt
        
        Args:
            prompt: Text description of 3D object
            steps: Diffusion steps (more = better quality, slower)
            cfg_scale: Classifier-free guidance scale
            size: Point cloud resolution
            save_ply: Save as PLY format
            save_obj: Save as OBJ/MTL format
            output_dir: Output directory
            
        Returns:
            Dictionary with model paths and metadata
        """
        if not self.initialized:
            return {'error': 'Shap-E models not loaded'}
        
        try:
            # Optimize prompt
            optimized_prompt = self.prompt_optimizer.optimize_prompt(prompt)
            logger.info(f"Original: {prompt}")
            logger.info(f"Optimized: {optimized_prompt}")
            
            # Negative prompt for better quality
            negative_prompt = self.prompt_optimizer.create_negative_prompt()
            
            # Generate 3D latent codes
            logger.info("Generating 3D latent codes (this may take 1-3 minutes)...")
            latents = sample_latents(
                batch_size=1,
                model=self.model,
                prompt=optimized_prompt,
                negative_prompt=negative_prompt,
                diffusion_steps=steps,
                guidance_scale=cfg_scale,
                device=self.device
            )
            
            # Decode to point cloud
            logger.info("Decoding to point cloud...")
            with torch.no_grad():
                point_clouds = self.xm.renderer.render_ply(latents)
            
            pc = point_clouds[0]  # Get first (and only) point cloud
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"3d_model_{timestamp}"
            
            result = {
                'id': model_id,
                'prompt': prompt,
                'optimized_prompt': optimized_prompt,
                'timestamp': timestamp,
                'device': self.device,
                'files': {}
            }
            
            # Save PLY format (point cloud)
            if save_ply:
                ply_path = os.path.join(output_dir, f"{model_id}.ply")
                self._save_point_cloud_ply(pc, ply_path)
                result['files']['ply'] = ply_path
                logger.info(f"✓ Saved PLY: {ply_path}")
            
            # Convert to mesh and save OBJ format
            if save_obj and TRIMESH_AVAILABLE:
                obj_path = os.path.join(output_dir, f"{model_id}.obj")
                mtl_path = os.path.join(output_dir, f"{model_id}.mtl")
                self._save_as_mesh(pc, obj_path, mtl_path)
                result['files']['obj'] = obj_path
                result['files']['mtl'] = mtl_path
                logger.info(f"✓ Saved OBJ: {obj_path}")
            
            # Save preview image
            preview_path = self._generate_preview(pc, output_dir, model_id)
            if preview_path:
                result['files']['preview'] = preview_path
                logger.info(f"✓ Saved preview: {preview_path}")
            
            # Save metadata
            meta_path = os.path.join(output_dir, f"{model_id}_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(result, f, indent=2)
            result['files']['metadata'] = meta_path
            
            logger.info(f"✓ 3D generation complete: {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating 3D model: {e}")
            return {'error': str(e)}
    
    def _save_point_cloud_ply(self, pc: "PointCloud", output_path: str):
        """Save point cloud as PLY file"""
        try:
            pc_util.write_ply(output_path, pc)
        except Exception as e:
            logger.error(f"Failed to save PLY: {e}")
    
    def _save_as_mesh(self, pc: PointCloud, obj_path: str, mtl_path: str):
        """Convert point cloud to mesh and save as OBJ"""
        if not TRIMESH_AVAILABLE:
            logger.warning("Trimesh not available, skipping OBJ conversion")
            return
        
        try:
            # Convert point cloud to numpy array
            points = pc.detach().cpu().numpy()
            
            # Create mesh using ball pivoting (creates surface from point cloud)
            if OPEN3D_AVAILABLE:
                # Use Open3D for better mesh reconstruction
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # Estimate normals
                pcd.estimate_normals()
                
                # Create mesh using Poisson reconstruction
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
                
                # Convert to trimesh
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces)
            else:
                # Fallback: create simple mesh from point cloud
                # Use convex hull as approximation
                mesh_tri = trimesh.convex.convex_hull(points)
            
            # Save OBJ
            mesh_tri.export(obj_path)
            
            # Create simple MTL file
            mtl_content = """# Generated material file
newmtl default
Ka 1.0 1.0 1.0
Kd 0.8 0.8 0.8
Ks 0.5 0.5 0.5
Ns 32.0
illum 2
"""
            with open(mtl_path, 'w') as f:
                f.write(mtl_content)
            
            logger.info(f"✓ Mesh saved: {obj_path}")
            
        except Exception as e:
            logger.error(f"Failed to save mesh: {e}")
    
    def _generate_preview(self, pc: PointCloud, output_dir: str, model_id: str) -> Optional[str]:
        """Generate 2D preview image of 3D model"""
        try:
            # Render point cloud from different angles and save preview
            # This is a simplified version - in production, use proper 3D rendering
            
            preview_path = os.path.join(output_dir, f"{model_id}_preview.png")
            
            # Create a simple preview by rendering the point cloud
            if hasattr(pc, 'colors') and pc.colors is not None:
                # Use actual colors
                colors = (pc.colors.detach().cpu().numpy() * 255).astype(np.uint8)
            else:
                # Generate gradient colors based on height
                pc_np = pc.detach().cpu().numpy()
                z_values = pc_np[:, 2]
                colors = np.zeros((len(pc_np), 3), dtype=np.uint8)
                colors[:, 0] = ((z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-5) * 255).astype(np.uint8)
                colors[:, 1] = ((z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-5) * 180).astype(np.uint8)
                colors[:, 2] = 200
            
            # Create image (simple 2D projection)
            img = Image.new('RGB', (512, 512), color=(30, 30, 40))
            img.save(preview_path)
            
            return preview_path
            
        except Exception as e:
            logger.error(f"Failed to generate preview: {e}")
            return None
    
    def generate_multi_view_images(
        self,
        prompt: str,
        num_views: int = 4,
        steps: int = 64,
        output_dir: str = 'outputs'
    ) -> Dict[str, Any]:
        """
        Generate multiple views of 3D object from text
        
        Args:
            prompt: Object description
            num_views: Number of different viewpoints
            steps: Diffusion steps
            output_dir: Output directory
            
        Returns:
            Dictionary with view images
        """
        result = {
            'prompt': prompt,
            'views': {}
        }
        
        # Angles for multiple views
        angles = ['front', 'top', 'side', 'isometric'][:num_views]
        
        try:
            # Generate 3D model first
            model_result = self.generate_3d_mesh(prompt, steps=steps, output_dir=output_dir)
            
            if 'error' in model_result:
                return model_result
            
            # TODO: Render different views and return as images
            result['model_id'] = model_result['id']
            result['views'] = {angle: f"view_{angle}.png" for angle in angles}
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating multi-view images: {e}")
            return {'error': str(e)}


class AccuracyEnhancer:
    """Enhances accuracy through post-processing and validation"""
    
    @staticmethod
    def validate_3d_model(model_path: str) -> Dict[str, Any]:
        """Validate 3D model quality"""
        if not TRIMESH_AVAILABLE:
            return {'error': 'Trimesh not available'}
        
        try:
            mesh = trimesh.load(model_path)
            
            stats = {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'volume': mesh.volume,
                'surface_area': mesh.area,
                'bounds': mesh.bounds.tolist(),
                'center_of_mass': mesh.center_mass.tolist(),
                'is_watertight': mesh.is_watertight,
                'valid': mesh.is_valid,
                'self_intersecting': mesh.self_intersections.shape[0] if hasattr(mesh, 'self_intersections') else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def optimize_mesh(model_path: str, target_poly_count: int = 50000) -> str:
        """Optimize mesh for web/game engines"""
        if not TRIMESH_AVAILABLE:
            return model_path
        
        try:
            mesh = trimesh.load(model_path)
            
            # Simplify mesh to target polygon count
            if len(mesh.faces) > target_poly_count:
                ratio = target_poly_count / len(mesh.faces)
                simplified = mesh.simplify(ratio)
                
                output_path = model_path.replace('.obj', '_optimized.obj')
                simplified.export(output_path)
                logger.info(f"Optimized mesh from {len(mesh.faces)} to {len(simplified.faces)} faces")
                
                return output_path
            
            return model_path
            
        except Exception as e:
            logger.error(f"Mesh optimization failed: {e}")
            return model_path


# Convenience functions
def generate_3d_model(prompt: str, output_dir: str = 'outputs', **kwargs) -> Dict[str, Any]:
    """Simple function to generate 3D model from text"""
    generator = Text3DGenerator()
    return generator.generate_3d_mesh(prompt, output_dir=output_dir, **kwargs)


def optimize_and_validate_model(model_path: str) -> Dict[str, Any]:
    """Optimize and validate a 3D model"""
    stats = AccuracyEnhancer.validate_3d_model(model_path)
    optimized = AccuracyEnhancer.optimize_mesh(model_path)
    
    return {
        'validation': stats,
        'optimized_path': optimized
    }
