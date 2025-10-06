import typer
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
import numpy as np

from src.utils.metrics import ssim, perceptual_proxy, iou, chamfer_distance, noise_score

app = typer.Typer(help="Compute quality metrics offline")


def pil_to_tensor(p: Path) -> torch.Tensor:
	 img = Image.open(p).convert("RGB")
	 t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
	 return t.unsqueeze(0)


@app.command()
def image_similarity(
	 ref: Path = typer.Option(..., exists=True),
	 pred: Path = typer.Option(..., exists=True),
):
	 a = pil_to_tensor(ref)
	 b = pil_to_tensor(pred)
	 print({
		 "ssim": float(ssim(a, b).item()),
		 "perceptual": float(perceptual_proxy(a, b).item()),
		 "noise_pred": float(noise_score(b).item()),
	 })


if __name__ == "__main__":
	 app()
