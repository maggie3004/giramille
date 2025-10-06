import os
import random
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
import typer
from rich import print

# Offline augmentations using OpenCV/NumPy only

def affine_transform(img: np.ndarray) -> np.ndarray:
	 h, w = img.shape[:2]
	 center = (w // 2, h // 2)
	 angle = random.uniform(-15, 15)
	 scale = random.uniform(0.9, 1.1)
	 M = cv2.getRotationMatrix2D(center, angle, scale)
	 tx = random.uniform(-0.05 * w, 0.05 * w)
	 ty = random.uniform(-0.05 * h, 0.05 * h)
	 M[:, 2] += [tx, ty]
	 return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def hue_shift(img: np.ndarray) -> np.ndarray:
	 hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	 shift = random.randint(-10, 10)
	 hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 180
	 return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def elastic_deform(img: np.ndarray, alpha: float = 34, sigma: float = 4) -> np.ndarray:
	 random_state = np.random.RandomState(None)
	 shape = img.shape[:2]
	 dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
	 dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
	 x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
	 map_x = (x + dx).astype(np.float32)
	 map_y = (y + dy).astype(np.float32)
	 return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def cutmix(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
	 h, w = img1.shape[:2]
	 cut_w = random.randint(w // 8, w // 3)
	 cut_h = random.randint(h // 8, h // 3)
	 x1 = random.randint(0, w - cut_w)
	 y1 = random.randint(0, h - cut_h)
	 patch = cv2.resize(img2, (cut_w, cut_h))
	 out = img1.copy()
	 out[y1:y1+cut_h, x1:x1+cut_w] = patch
	 return out

def load_images_from_dir(data_dir: Path) -> List[Path]:
	 exts = {'.png', '.jpg', '.jpeg', '.bmp'}
	 files = []
	 for root, _, arr in os.walk(data_dir):
		 for f in arr:
			 if Path(f).suffix.lower() in exts:
				 files.append(Path(root) / f)
	 return files

app = typer.Typer(help="Prepare local dataset: augment + split train/val/test")

@app.command()
def main(
	 data_dir: Path = typer.Option(..., "--data-dir", "--data_dir", help="Path to local dataset root", dir_okay=True, file_okay=False, exists=True, readable=True, resolve_path=True, writable=False, allow_dash=False, path_type=Path),
	 out_dir: Path = typer.Option(Path("data/augmented"), "--out-dir", "--out_dir", help="Output directory for augmented dataset"),
	 val_ratio: float = typer.Option(0.1, "--val-ratio", "--val_ratio", min=0.0, max=0.5),
	 test_ratio: float = typer.Option(0.1, "--test-ratio", "--test_ratio", min=0.0, max=0.5),
	 image_size: int = typer.Option(256, "--image-size", "--image_size", help="Resize square size"),
	 augmentations: List[str] = typer.Option(["affine", "hue", "elastic", "cutmix"], "--augmentations", help="Augmentations to apply"),
	 copies_per_image: int = typer.Option(2, "--copies-per-image", "--copies_per_image", help="Augmented copies per source image"),
):
	 random.seed(42)
	 np.random.seed(42)
	 files = load_images_from_dir(data_dir)
	 if len(files) == 0:
		 print("[red]No images found in dataset dir[/red]")
		 raise typer.Exit(code=1)

	 out_images = []
	 out_dir.mkdir(parents=True, exist_ok=True)

	 for idx, path in enumerate(files):
		 img = cv2.imread(str(path), cv2.IMREAD_COLOR)
		 if img is None:
			 continue
		 img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
		 base_out = out_dir / f"{path.stem}_base_{idx}.png"
		 cv2.imwrite(str(base_out), img)
		 out_images.append(base_out)

		 for c in range(copies_per_image):
			 aug = img.copy()
			 if "affine" in augmentations:
				 aug = affine_transform(aug)
			 if "hue" in augmentations:
				 aug = hue_shift(aug)
			 if "elastic" in augmentations:
				 aug = elastic_deform(aug)
			 if "cutmix" in augmentations and len(files) > 1:
				 other = files[(idx + c + 1) % len(files)]
				 oimg = cv2.imread(str(other), cv2.IMREAD_COLOR)
				 if oimg is not None:
					 oimg = cv2.resize(oimg, (image_size, image_size), interpolation=cv2.INTER_AREA)
					 aug = cutmix(aug, oimg)
			 out_path = out_dir / f"{path.stem}_aug{c}_{idx}.png"
			 cv2.imwrite(str(out_path), aug)
			 out_images.append(out_path)

	 # split
	 random.shuffle(out_images)
	 n = len(out_images)
	 n_test = int(n * test_ratio)
	 n_val = int(n * val_ratio)
	 test_set = out_images[:n_test]
	 val_set = out_images[n_test:n_test+n_val]
	 train_set = out_images[n_test+n_val:]

	 for split_name, split in [("train", train_set), ("val", val_set), ("test", test_set)]:
		 split_dir = out_dir / split_name
		 split_dir.mkdir(parents=True, exist_ok=True)
		 for p in split:
			 shutil.move(str(p), split_dir / p.name)

	 print(f"[green]Done[/green]: train={len(train_set)} val={len(val_set)} test={len(test_set)} -> {out_dir}")

if __name__ == "__main__":
	 app()
