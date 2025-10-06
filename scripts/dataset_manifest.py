import hashlib
import json
from pathlib import Path
from typing import List
import typer

app = typer.Typer(help="Generate dataset manifest with checksums and split lists")


def sha256_file(p: Path) -> str:
	 h = hashlib.sha256()
	 with open(p, 'rb') as f:
		 for chunk in iter(lambda: f.read(1024 * 1024), b''):
			 h.update(chunk)
	 return h.hexdigest()


@app.command()
def main(
	 data_dir: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False, resolve_path=True),
	 out_json: Path = typer.Option(Path('outputs/dataset_manifest.json')),
):
	 exts = {'.png', '.jpg', '.jpeg', '.bmp'}
	 files: List[Path] = []
	 for p in data_dir.rglob('*'):
		 if p.suffix.lower() in exts:
			 files.append(p)
	 files = sorted(files)
	 entries = []
	 for p in files:
		 entries.append({
			 'path': str(p.relative_to(data_dir)),
			 'sha256': sha256_file(p)
		 })
	 manifest = {
		 'root': str(data_dir),
		 'count': len(entries),
		 'files': entries,
	 }
	 out_json.parent.mkdir(parents=True, exist_ok=True)
	 out_json.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
	 typer.echo(f"Wrote manifest: {out_json} ({len(entries)} files)")


if __name__ == '__main__':
	 app()
