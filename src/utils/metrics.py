from typing import Tuple
import math
import torch
import torch.nn.functional as F

# SSIM (single-scale)

def ssim(img1: torch.Tensor, img2: torch.Tensor, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> torch.Tensor:
	 # expects NCHW in [0,1]
	 mu1 = F.avg_pool2d(img1, 3, 1, 1)
	 mu2 = F.avg_pool2d(img2, 3, 1, 1)
	 mu1_sq = mu1.pow(2)
	 mu2_sq = mu2.pow(2)
	 mu1_mu2 = mu1 * mu2
	 sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
	 sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
	 sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
	 ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
	 return ssim_map.mean()

# Perceptual proxy: MS-SSIM + gradient magnitude similarity

def perceptual_proxy(img1: torch.Tensor, img2: torch.Tensor, levels: int = 3) -> torch.Tensor:
	 def grad_mag(x):
		 gx = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:] - F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]
		 gy = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :] - F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]
		 return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
	 weights = [0.5, 0.3, 0.2]
	 assert len(weights) == levels
	 score = 0.0
	 x1, x2 = img1, img2
	 for i in range(levels):
		 score += weights[i] * ssim(x1, x2)
		 g1, g2 = grad_mag(x1), grad_mag(x2)
		 gsim = (2 * g1 * g2 + 1e-4) / (g1 ** 2 + g2 ** 2 + 1e-4)
		 score += weights[i] * gsim.mean()
		 if i < levels - 1:
			 x1 = F.avg_pool2d(x1, 2)
			 x2 = F.avg_pool2d(x2, 2)
	 return score / (2 * sum(weights))

# Vector IoU and Chamfer (raster proxy input: masks)

def iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
	 inter = (mask1 & mask2).float().sum()
	 union = (mask1 | mask2).float().sum().clamp_min(1.0)
	 return inter / union


def chamfer_distance(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
	 # points: [N,2], [M,2]
	 if points1.numel() == 0 or points2.numel() == 0:
		 return torch.tensor(1.0)
	 d1 = ((points1.unsqueeze(1) - points2.unsqueeze(0)) ** 2).sum(-1)
	 cd = d1.min(dim=1)[0].mean() + d1.min(dim=0)[0].mean()
	 return cd

# Noise/artifact detector: high-frequency energy ratio

def noise_score(img: torch.Tensor) -> torch.Tensor:
	 # simple Laplacian variance normalized by intensity variance
	 gray = img.mean(dim=1, keepdim=True)
	 k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=img.dtype, device=img.device).view(1,1,3,3)
	 lap = F.conv2d(gray, k, padding=1)
	 score = lap.var() / (gray.var() + 1e-6)
	 return score
