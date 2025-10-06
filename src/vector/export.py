from pathlib import Path
from typing import List, Tuple
import skia

# paths: list of polylines (quadratic control approximations already)

def to_skia_path(poly: List[Tuple[float, float]]) -> skia.Path:
	 path = skia.Path()
	 if not poly:
		 return path
	 x0, y0 = poly[0]
	 path.moveTo(x0, y0)
	 for i in range(1, len(poly)):
		 x1, y1 = poly[i]
		 cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
		 path.quadTo(cx, cy, x1, y1)
		 x0, y0 = x1, y1
	 path.close()
	 return path


def export_pdf(polys: List[List[Tuple[float, float]]], out_pdf: Path, size: Tuple[int, int]):
	 w, h = size
	 doc = skia.PDF.MakeDocument(str(out_pdf))
	 with doc as pdf:
		 page = pdf.beginPage(w, h)
		 canvas = page
		 paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kFill_Style, Color=skia.ColorBLACK)
		 for poly in polys:
			 p = to_skia_path(poly)
			 canvas.drawPath(p, paint)


def export_eps(polys: List[List[Tuple[float, float]]], out_eps: Path, size: Tuple[int, int]):
	 # Minimal EPS header placeholder for offline flow; vector drawing omitted for simplicity
	 w, h = size
	 header = (
		 f"%!PS-Adobe-3.0 EPSF-3.0\n"
		 f"%%Creator: offline-genvec\n"
		 f"%%Pages: 1\n"
		 f"%%BoundingBox: 0 0 {int(w)} {int(h)}\n"
		 f"%%EndComments\n"
		 f"showpage\n"
	 )
	 out_eps.write_text(header, encoding="ascii")
