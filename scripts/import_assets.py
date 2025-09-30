import shutil
from pathlib import Path
import typer

app = typer.Typer(help="Import front-end PNG assets into ./static with expected names")

EXPECTED = {
	 'bg': 'bg.png',
	 'btn_gen_vector': 'btn_gen_vector.png',
	 'btn_gen_png': 'btn_gen_png.png',
	 'btn_export': 'btn_export.png',
	 'btn_resize': 'btn_resize.png',
	 'btn_retouch': 'btn_retouch.png',
	 'btn_cancel': 'btn_cancel.png',
}

# Synonym keywords (lowercase, matched against filename stem)
SYNONYMS = {
	 'bg': ['bg', 'background', 'fundo', 'tela principal'],
	 'btn_gen_vector': ['gerar vetor', 'vector', 'vetor'],
	 'btn_gen_png': ['gerar png', 'png', 'gerar imagem'],
	 'btn_export': ['export', 'exportar'],
	 'btn_resize': ['resize', 'redimensionar', 'redimencionar'],
	 'btn_retouch': ['retouch', 'retocar'],
	 'btn_cancel': ['cancel', 'cancelar'],
}


def choose_candidate(candidates: list[Path], key: str) -> Path | None:
	 # prefer exact expected name
	 for c in candidates:
		 if c.name.lower() == EXPECTED[key].lower():
			 return c
	 # match by synonyms on stem
	 words = SYNONYMS.get(key, [])
	 for c in candidates:
		 stem = c.stem.lower()
		 if any(w in stem for w in words):
			 return c
	 return None


@app.command()
def main(
	 src_dir: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False, resolve_path=True),
):
	 dst = Path('static')
	 dst.mkdir(parents=True, exist_ok=True)
	 candidates = list(src_dir.rglob('*.png'))
	 if not candidates:
		 typer.echo("No PNG files found in source directory.")
		 raise typer.Exit(code=1)

	 for key, dst_name in EXPECTED.items():
		 chosen = choose_candidate(candidates, key)
		 if chosen is None:
			 typer.echo(f"[skip] {key} -> static/{dst_name} (no match found)")
			 continue
		 shutil.copyfile(chosen, dst / dst_name)
		 typer.echo(f"[ok] {key} -> static/{dst_name} (from {chosen.name})")

	 typer.echo("Done. Launch UI: python ui.py")


if __name__ == '__main__':
	 app()
