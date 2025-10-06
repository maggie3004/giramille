import sys
import subprocess
from pathlib import Path
from typing import Optional
import typer

app = typer.Typer(help="Run the full offline pipeline with one command")


def run(cmd: list[str]):
	 print(f"[run] {' '.join(cmd)}")
	 subprocess.run(cmd, check=True)


@app.command()
def all(
	 data_dir: Path = typer.Option(..., exists=True, file_ok=False, dir_ok=True, help="Path to local dataset root"),
	 out_data: Path = typer.Option(Path("data/augmented"), help="Augmented dataset output"),
	 image_size: int = typer.Option(256, help="Training image size"),
	 val_ratio: float = typer.Option(0.1),
	 test_ratio: float = typer.Option(0.1),
	 copies_per_image: int = typer.Option(2),
	 # training
	 batch_size: int = typer.Option(16),
	 max_steps: int = typer.Option(100000),
	 gpus: int = typer.Option(2),
	 mixed_precision: bool = typer.Option(True),
	 # inference
	 num_images: int = typer.Option(8),
	 # vectorization
	 max_layers: int = typer.Option(20),
	 max_anchors: int = typer.Option(300),
	 # UI
	 run_ui: bool = typer.Option(False, help="Launch the local UI server at the end"),
):
	 py = sys.executable

	 # 1) Prepare dataset
	 run([
		 py, "scripts/prepare_dataset.py",
		 "--data_dir", str(data_dir),
		 "--out_dir", str(out_data),
		 "--val_ratio", str(val_ratio),
		 "--test_ratio", str(test_ratio),
		 "--image_size", str(image_size),
		 "--copies_per_image", str(copies_per_image),
	 ])

	 # 2) Train diffusion model (Hydra overrides inline)
	 run([
		 py, "train.py",
		 f"hydra.run.dir=outputs/train",
		 f"trainer.batch_size={batch_size}",
		 f"trainer.max_steps={max_steps}",
		 f"trainer.gpus={gpus}",
		 f"trainer.mixed_precision={'true' if mixed_precision else 'false'}",
		 f"trainer.image_size={image_size}",
		 f"paths.data_root={out_data}",
	 ])

	 # 3) Inference (samples)
	 checkpoint = Path("outputs/checkpoints/last.pt")
	 run([
		 py, "infer.py",
		 "--checkpoint", str(checkpoint),
		 "--out_dir", "outputs/samples",
		 "--num_images", str(num_images),
		 "--image_size", str(image_size),
	 ])

	 # 4) Vectorization
	 run([
		 py, "vectorize.py",
		 "--input_dir", "outputs/samples",
		 "--out_dir", "outputs/vectors",
		 "--image_size", str(image_size),
		 "--max_layers", str(max_layers),
		 "--max_anchors", str(max_anchors),
	 ])

	 # 5) Optional UI
	 if run_ui:
		 print("Launching UI at http://127.0.0.1:5000 ... (Ctrl+C to stop)")
		 subprocess.run([py, "ui.py"])  # foreground


if __name__ == "__main__":
	 app()
