<<<<<<< HEAD
﻿import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.diffusion_unet import UNetModel
from src.data.dataset import ImageFolderDataset
from src.utils.checkpoint import save_checkpoint
from src.utils.distributed import setup_ddp, cleanup_ddp, is_distributed, get_rank, is_main_process


def get_dataloader(data_root: Path, split: str, image_size: int, batch_size: int, num_workers: int = 4):
	 tfm = transforms.Compose([
		 transforms.Resize((image_size, image_size)),
		 transforms.ToTensor(),
	 ])
	 ds = ImageFolderDataset(data_root, split=split, transform=tfm)
	 sampler = None
	 if is_distributed():
		 from torch.utils.data.distributed import DistributedSampler
		 sampler = DistributedSampler(ds, shuffle=True)
	 loader = DataLoader(ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
	 return loader


def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 2e-2):
	 return torch.linspace(start, end, timesteps)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
	 torch.manual_seed(cfg.seed)
	 setup_ddp()
	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	 model = UNetModel(in_channels=3, base_channels=cfg.trainer.channels, channel_mults=cfg.trainer.channel_mults)
	 model.to(device)
	 if is_distributed():
		 local_rank = int(os.environ.get("LOCAL_RANK", 0))
		 model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

	 optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.lr)
	 scaler = torch.cuda.amp.GradScaler(enabled=cfg.trainer.mixed_precision)

	 betas = linear_beta_schedule(cfg.trainer.timesteps).to(device)
	 alphas = 1.0 - betas
	 alphas_cumprod = torch.cumprod(alphas, dim=0)

	 def q_sample(x0, t, noise):
		 sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
		 sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
		 return sqrt_alphas_cumprod * x0 + sqrt_one_minus * noise

	 loader = get_dataloader(Path(cfg.paths.data_root), "train", cfg.trainer.image_size, cfg.trainer.batch_size)

	 global_step = 0
	 model.train()
	 for epoch in range(10**9):  # loop until max_steps
		 for x, _ in loader:
			 x = x.to(device)
			 t = torch.randint(0, cfg.trainer.timesteps, (x.size(0),), device=device)
			 noise = torch.randn_like(x)
			 x_noisy = q_sample(x, t, noise)

			 optimizer.zero_grad(set_to_none=True)
			 with torch.cuda.amp.autocast(enabled=cfg.trainer.mixed_precision):
				 pred = model(x_noisy, t)
				 loss = torch.mean((pred - noise) ** 2)
			 scaler.scale(loss).backward()
			 if cfg.trainer.grad_checkpoint:
				 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			 scaler.step(optimizer)
			 scaler.update()

			 if is_main_process() and (global_step % cfg.trainer.log_interval == 0):
				 print({"step": global_step, "loss": float(loss.item())})
			 if is_main_process() and (global_step % cfg.trainer.ckpt_interval == 0 and global_step > 0):
				 out = Path(cfg.paths.checkpoints)
				 out.mkdir(parents=True, exist_ok=True)
				 save_checkpoint({"model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
								  "optimizer": optimizer.state_dict(), "step": global_step}, out / f"step_{global_step}.pt")
				 save_checkpoint({"model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
								  "optimizer": optimizer.state_dict(), "step": global_step}, out / "last.pt")

			 global_step += 1
			 if global_step >= cfg.trainer.max_steps:
				 break
		 if global_step >= cfg.trainer.max_steps:
			 break

	 cleanup_ddp()

if __name__ == "__main__":
	 main()
=======
﻿import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.diffusion_unet import UNetModel
from src.data.dataset import ImageFolderDataset
from src.utils.checkpoint import save_checkpoint
from src.utils.distributed import setup_ddp, cleanup_ddp, is_distributed, get_rank, is_main_process


def get_dataloader(data_root: Path, split: str, image_size: int, batch_size: int, num_workers: int = 4):
	 tfm = transforms.Compose([
		 transforms.Resize((image_size, image_size)),
		 transforms.ToTensor(),
	 ])
	 ds = ImageFolderDataset(data_root, split=split, transform=tfm)
	 sampler = None
	 if is_distributed():
		 from torch.utils.data.distributed import DistributedSampler
		 sampler = DistributedSampler(ds, shuffle=True)
	 loader = DataLoader(ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
	 return loader


def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 2e-2):
	 return torch.linspace(start, end, timesteps)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
	 torch.manual_seed(cfg.seed)
	 setup_ddp()
	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	 model = UNetModel(in_channels=3, base_channels=cfg.trainer.channels, channel_mults=cfg.trainer.channel_mults)
	 model.to(device)
	 if is_distributed():
		 local_rank = int(os.environ.get("LOCAL_RANK", 0))
		 model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

	 optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.lr)
	 scaler = torch.cuda.amp.GradScaler(enabled=cfg.trainer.mixed_precision)

	 betas = linear_beta_schedule(cfg.trainer.timesteps).to(device)
	 alphas = 1.0 - betas
	 alphas_cumprod = torch.cumprod(alphas, dim=0)

	 def q_sample(x0, t, noise):
		 sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
		 sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
		 return sqrt_alphas_cumprod * x0 + sqrt_one_minus * noise

	 loader = get_dataloader(Path(cfg.paths.data_root), "train", cfg.trainer.image_size, cfg.trainer.batch_size)

	 global_step = 0
	 model.train()
	 for epoch in range(10**9):  # loop until max_steps
		 for x, _ in loader:
			 x = x.to(device)
			 t = torch.randint(0, cfg.trainer.timesteps, (x.size(0),), device=device)
			 noise = torch.randn_like(x)
			 x_noisy = q_sample(x, t, noise)

			 optimizer.zero_grad(set_to_none=True)
			 with torch.cuda.amp.autocast(enabled=cfg.trainer.mixed_precision):
				 pred = model(x_noisy, t)
				 loss = torch.mean((pred - noise) ** 2)
			 scaler.scale(loss).backward()
			 if cfg.trainer.grad_checkpoint:
				 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			 scaler.step(optimizer)
			 scaler.update()

			 if is_main_process() and (global_step % cfg.trainer.log_interval == 0):
				 print({"step": global_step, "loss": float(loss.item())})
			 if is_main_process() and (global_step % cfg.trainer.ckpt_interval == 0 and global_step > 0):
				 out = Path(cfg.paths.checkpoints)
				 out.mkdir(parents=True, exist_ok=True)
				 save_checkpoint({"model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
								  "optimizer": optimizer.state_dict(), "step": global_step}, out / f"step_{global_step}.pt")
				 save_checkpoint({"model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
								  "optimizer": optimizer.state_dict(), "step": global_step}, out / "last.pt")

			 global_step += 1
			 if global_step >= cfg.trainer.max_steps:
				 break
		 if global_step >= cfg.trainer.max_steps:
			 break

	 cleanup_ddp()

if __name__ == "__main__":
	 main()
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
