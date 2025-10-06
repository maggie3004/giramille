from pathlib import Path
from typing import Callable, Optional, List, Tuple
import glob

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
	 def __init__(self, root: Path, split: str = "train", transform: Optional[Callable] = None):
		 self.root = Path(root)
		 self.split = split
		 self.dir = self.root / split
		 self.transform = transform
		 self.files: List[str] = []
		 exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
		 for e in exts:
			 self.files.extend(glob.glob(str(self.dir / e)))
		 if len(self.files) == 0:
			 raise RuntimeError(f"No images found in {self.dir}")

	 def __len__(self) -> int:
		 return len(self.files)

	 def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
		 path = self.files[idx]
		 img = Image.open(path).convert("RGB")
		 if self.transform is not None:
			 img = self.transform(img)
		 else:
			 img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
		 return img, {"path": path}
