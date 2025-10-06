<<<<<<< HEAD
﻿from pathlib import Path
from typing import List
import typer
import torch
import cv2
import numpy as np

from src.models.segnet import SmallUNet
from src.vector.curve_fit import contours_to_beziers, beziers_to_svg
from src.vector.postprocess import reduce_anchors, merge_layers
from src.vector.export import export_pdf, export_eps

app = typer.Typer(help="Convert raster images to layered vector formats")


@app.command()
def main(
	 input_dir: Path = typer.Option(..., exists=True, file_ok=False, dir_ok=True),
	 out_dir: Path = typer.Option(Path("outputs/vectors")),
	 image_size: int = typer.Option(256),
	 num_classes: int = typer.Option(16),
	 max_layers: int = typer.Option(20),
	 max_anchors: int = typer.Option(300),
):
	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	 model = SmallUNet(in_ch=3, num_classes=num_classes).to(device)
	 model.eval()  # randomly initialized; replace with trained checkpoint for better segmentation

	 out_dir.mkdir(parents=True, exist_ok=True)
	 exts = {".png", ".jpg", ".jpeg", ".bmp"}
	 files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
	 for p in files:
		 img = cv2.imread(str(p), cv2.IMREAD_COLOR)
		 if img is None:
			 continue
		 h0, w0 = img.shape[:2]
		 img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
		 x = torch.from_numpy(img_resized[:, :, ::-1]).float().permute(2, 0, 1) / 255.0
		 x = x.unsqueeze(0).to(device)

		 with torch.no_grad():
			 logits = model(x)
			 mask = logits.argmax(dim=1)[0].byte().cpu().numpy()

		 layer_paths: List[List[List[tuple]]] = []
		 for cls in range(num_classes):
			 cls_mask = (mask == cls).astype(np.uint8) * 255
			 if cls_mask.sum() < 10:
				 continue
			 paths = contours_to_beziers(cls_mask, epsilon=1.5, max_segments=8)
			 paths = reduce_anchors(paths, max_anchors=max_anchors)
			 layer_paths.append(paths)

		 merged = merge_layers(layer_paths, max_layers=max_layers)
		 svg_out = out_dir / f"{p.stem}.svg"
		 beziers_to_svg(merged, svg_out, size=(w0, h0))

		 pdf_out = out_dir / f"{p.stem}.pdf"
		 export_pdf(merged, pdf_out, size=(w0, h0))

		 eps_out = out_dir / f"{p.stem}.eps"
		 export_eps(merged, eps_out, size=(w0, h0))

		 print(f"Vector exported: {svg_out}, {pdf_out}, {eps_out}")

if __name__ == "__main__":
	 app()
=======
﻿from pathlib import Path
from typing import List
import typer
import torch
import cv2
import numpy as np

from src.models.segnet import SmallUNet
from src.vector.curve_fit import contours_to_beziers, beziers_to_svg
from src.vector.postprocess import reduce_anchors, merge_layers
from src.vector.export import export_pdf, export_eps

app = typer.Typer(help="Convert raster images to layered vector formats")


@app.command()
def main(
	 input_dir: Path = typer.Option(..., exists=True, file_ok=False, dir_ok=True),
	 out_dir: Path = typer.Option(Path("outputs/vectors")),
	 image_size: int = typer.Option(256),
	 num_classes: int = typer.Option(16),
	 max_layers: int = typer.Option(20),
	 max_anchors: int = typer.Option(300),
):
	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	 model = SmallUNet(in_ch=3, num_classes=num_classes).to(device)
	 model.eval()  # randomly initialized; replace with trained checkpoint for better segmentation

	 out_dir.mkdir(parents=True, exist_ok=True)
	 exts = {".png", ".jpg", ".jpeg", ".bmp"}
	 files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
	 for p in files:
		 img = cv2.imread(str(p), cv2.IMREAD_COLOR)
		 if img is None:
			 continue
		 h0, w0 = img.shape[:2]
		 img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
		 x = torch.from_numpy(img_resized[:, :, ::-1]).float().permute(2, 0, 1) / 255.0
		 x = x.unsqueeze(0).to(device)

		 with torch.no_grad():
			 logits = model(x)
			 mask = logits.argmax(dim=1)[0].byte().cpu().numpy()

		 layer_paths: List[List[List[tuple]]] = []
		 for cls in range(num_classes):
			 cls_mask = (mask == cls).astype(np.uint8) * 255
			 if cls_mask.sum() < 10:
				 continue
			 paths = contours_to_beziers(cls_mask, epsilon=1.5, max_segments=8)
			 paths = reduce_anchors(paths, max_anchors=max_anchors)
			 layer_paths.append(paths)

		 merged = merge_layers(layer_paths, max_layers=max_layers)
		 svg_out = out_dir / f"{p.stem}.svg"
		 beziers_to_svg(merged, svg_out, size=(w0, h0))

		 pdf_out = out_dir / f"{p.stem}.pdf"
		 export_pdf(merged, pdf_out, size=(w0, h0))

		 eps_out = out_dir / f"{p.stem}.eps"
		 export_eps(merged, eps_out, size=(w0, h0))

		 print(f"Vector exported: {svg_out}, {pdf_out}, {eps_out}")

if __name__ == "__main__":
	 app()
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
