import json
from pathlib import Path
from typing import Dict, List
import statistics as stats
import typer
import xml.etree.ElementTree as ET
from svgpathtools import svg2paths2

app = typer.Typer(help="Analyze reference SVG vectors to derive recommended max_layers and max_anchors")


def count_groups(svg_path: Path) -> int:
	 try:
		 tree = ET.parse(svg_path)
		 root = tree.getroot()
		 ns = ''
		 if root.tag.startswith('{'):
			 ns = root.tag.split('}')[0] + '}'
		 groups = root.findall('.//' + ns + 'g')
		 return len(groups)
	 except Exception:
		 return 0


def anchors_per_path(svg_path: Path) -> List[int]:
	 paths, attrs, svg_attr = svg2paths2(str(svg_path))
	 counts = []
	 for p in paths:
		 # approximate anchors = number of segment end points
		 counts.append(len(p))
	 return counts


@app.command()
def main(
	 ref_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True),
	 out_json: Path = typer.Option(Path('outputs/reference_stats.json')),
):
	 svgs = [p for p in ref_dir.rglob('*.svg')]
	 if not svgs:
		 typer.echo("No SVG files found in reference directory.")
		 raise typer.Exit(code=1)

	 layer_counts: List[int] = []
	 per_path_anchor_counts: List[int] = []
	 total_paths: int = 0

	 for p in svgs:
		 layer_counts.append(count_groups(p))
		 anchors = anchors_per_path(p)
		 per_path_anchor_counts.extend(anchors)
		 total_paths += len(anchors)

	 def percentile(arr: List[int], q: float) -> int:
		 if not arr:
			 return 0
		 arr_sorted = sorted(arr)
		 k = int(round((q / 100.0) * (len(arr_sorted) - 1)))
		 return int(arr_sorted[k])

	 rec_max_layers = max(5, percentile(layer_counts, 90)) if layer_counts else 20
	 rec_max_anchors = max(20, percentile(per_path_anchor_counts, 90)) if per_path_anchor_counts else 300

	 report: Dict = {
		 'files_analyzed': len(svgs),
		 'total_paths': total_paths,
		 'layer_counts_summary': {
			 'min': min(layer_counts) if layer_counts else 0,
			 'median': stats.median(layer_counts) if layer_counts else 0,
			 'p90': percentile(layer_counts, 90) if layer_counts else 0,
			 'max': max(layer_counts) if layer_counts else 0,
		 },
		 'anchors_per_path_summary': {
			 'min': min(per_path_anchor_counts) if per_path_anchor_counts else 0,
			 'median': stats.median(per_path_anchor_counts) if per_path_anchor_counts else 0,
			 'p90': percentile(per_path_anchor_counts, 90) if per_path_anchor_counts else 0,
			 'max': max(per_path_anchor_counts) if per_path_anchor_counts else 0,
		 },
		 'recommended': {
			 'max_layers': rec_max_layers,
			 'max_anchors': rec_max_anchors,
		 }
	 }

	 out_json.parent.mkdir(parents=True, exist_ok=True)
	 out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
	 typer.echo(f"Saved analysis to {out_json}")
	 typer.echo(f"Recommended: --max_layers {rec_max_layers} --max_anchors {rec_max_anchors}")


if __name__ == '__main__':
	 app()
