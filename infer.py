<<<<<<< HEAD
﻿from pathlib import Path
import math
import typer
import torch
from torchvision.utils import save_image

from src.models.diffusion_unet import UNetModel
from src.utils.checkpoint import load_checkpoint

app = typer.Typer(help="Generate images from scratch using the trained diffusion model")


def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 2e-2):
	 return torch.linspace(start, end, timesteps)


@app.command()
def main(
	 checkpoint: Path = typer.Option(..., exists=True),
	 out_dir: Path = typer.Option(Path("outputs/samples")),
	 num_images: int = typer.Option(8),
	 image_size: int = typer.Option(256),
	 timesteps: int = typer.Option(1000),
	 steps: int = typer.Option(50),
	 channels: int = typer.Option(64),
):
	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	 state = load_checkpoint(checkpoint, map_location=device)
	 model = UNetModel(in_channels=3, base_channels=channels).to(device)
	 model.load_state_dict(state["model"])
	 model.eval()

	 betas = linear_beta_schedule(timesteps).to(device)
	 alphas = 1.0 - betas
	 alphas_cumprod = torch.cumprod(alphas, dim=0)
	 alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
	 posterior_var = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

	 def p_sample(x, t_idx):
		 with torch.no_grad():
			 pred_noise = model(x, torch.full((x.size(0),), t_idx, device=device, dtype=torch.long))
			 alpha_t = alphas[t_idx]
			 alpha_bar_t = alphas_cumprod[t_idx]
			 sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
			 sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
			 x0_pred = (x - sqrt_one_minus * pred_noise) / torch.sqrt(alpha_bar_t)
			 dir_xt = torch.sqrt(1 - alpha_t) * pred_noise
			 noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
			 var = torch.sqrt(posterior_var[t_idx]) * noise
			 x = sqrt_recip_alpha * (x - dir_xt) + var
			 return x

	 out_dir.mkdir(parents=True, exist_ok=True)
	 x = torch.randn(num_images, 3, image_size, image_size, device=device)
	 t_list = torch.linspace(timesteps - 1, 0, steps, dtype=torch.long)
	 for t_idx in t_list:
		 x = p_sample(x, int(t_idx))

	 x = (x.clamp(-1, 1) + 1) / 2.0
	 for i in range(num_images):
		 save_image(x[i], out_dir / f"sample_{i:03d}.png")
	 print(f"Saved {num_images} images to {out_dir}")

if __name__ == "__main__":
	 app()
=======
﻿from pathlib import Path
import math
import typer
import torch
from torchvision.utils import save_image

from src.models.diffusion_unet import UNetModel
from src.utils.checkpoint import load_checkpoint

app = typer.Typer(help="Generate images from scratch using the trained diffusion model")


def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 2e-2):
	 return torch.linspace(start, end, timesteps)


@app.command()
def main(
	 checkpoint: Path = typer.Option(..., exists=True),
	 out_dir: Path = typer.Option(Path("outputs/samples")),
	 num_images: int = typer.Option(8),
	 image_size: int = typer.Option(256),
	 timesteps: int = typer.Option(1000),
	 steps: int = typer.Option(50),
	 channels: int = typer.Option(64),
):
	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	 state = load_checkpoint(checkpoint, map_location=device)
	 model = UNetModel(in_channels=3, base_channels=channels).to(device)
	 model.load_state_dict(state["model"])
	 model.eval()

	 betas = linear_beta_schedule(timesteps).to(device)
	 alphas = 1.0 - betas
	 alphas_cumprod = torch.cumprod(alphas, dim=0)
	 alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
	 posterior_var = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

	 def p_sample(x, t_idx):
		 with torch.no_grad():
			 pred_noise = model(x, torch.full((x.size(0),), t_idx, device=device, dtype=torch.long))
			 alpha_t = alphas[t_idx]
			 alpha_bar_t = alphas_cumprod[t_idx]
			 sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
			 sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
			 x0_pred = (x - sqrt_one_minus * pred_noise) / torch.sqrt(alpha_bar_t)
			 dir_xt = torch.sqrt(1 - alpha_t) * pred_noise
			 noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
			 var = torch.sqrt(posterior_var[t_idx]) * noise
			 x = sqrt_recip_alpha * (x - dir_xt) + var
			 return x

	 out_dir.mkdir(parents=True, exist_ok=True)
	 x = torch.randn(num_images, 3, image_size, image_size, device=device)
	 t_list = torch.linspace(timesteps - 1, 0, steps, dtype=torch.long)
	 for t_idx in t_list:
		 x = p_sample(x, int(t_idx))

	 x = (x.clamp(-1, 1) + 1) / 2.0
	 for i in range(num_images):
		 save_image(x[i], out_dir / f"sample_{i:03d}.png")
	 print(f"Saved {num_images} images to {out_dir}")

if __name__ == "__main__":
	 app()
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
