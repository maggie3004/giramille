﻿from pathlib import Path
from typing import Any, Dict
import torch


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
	 path = Path(path)
	 path.parent.mkdir(parents=True, exist_ok=True)
	 torch.save(state, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
	 return torch.load(Path(path), map_location=map_location)
