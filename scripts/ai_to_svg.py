import subprocess
from pathlib import Path
import typer

app = typer.Typer(help="Convert .ai files to .svg using Inkscape CLI (offline)")


@app.command()
def main(
	 src_dir: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False, resolve_path=True),
	 out_dir: Path = typer.Option(Path('references/vectors'), dir_okay=True, file_okay=False),
	 inkscape: Path = typer.Option(Path(r"C:\Program Files\Inkscape\bin\inkscape.exe"), help="Path to inkscape.exe"),
):
	 out_dir.mkdir(parents=True, exist_ok=True)
	 files = [p for p in src_dir.rglob('*.ai')]
	 if not files:
		 typer.echo("No .ai files found.")
		 raise typer.Exit(code=1)
	 for f in files:
		 out_svg = out_dir / (f.stem + '.svg')
		 cmd = [str(inkscape), str(f), '--export-plain-svg', str(out_svg)]
		 typer.echo(" ".join(cmd))
		 subprocess.run(cmd, check=True)
	 typer.echo(f"Done. SVGs at {out_dir}")


if __name__ == '__main__':
	 app()
