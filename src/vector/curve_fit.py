from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import cv2
from svgpathtools import Path as SVGPath, Line, CubicBezier, QuadraticBezier, wsvg

# Placeholder: in this POC we approximate contours with bezier curves via approxPolyDP + simple smoothing

def contours_to_beziers(mask: np.ndarray, epsilon: float = 1.5, max_segments: int = 8) -> List[List[Tuple[float, float]]]:
	 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	 paths = []
	 for cnt in contours:
		 if len(cnt) < 3:
			 continue
		 approx = cv2.approxPolyDP(cnt, epsilon, True)
		 pts = [(float(p[0][0]), float(p[0][1])) for p in approx][: max_segments]
		 if len(pts) >= 3:
			 paths.append(pts)
	 return paths


def beziers_to_svg(paths: List[List[Tuple[float, float]]], out_file: Path, size: Tuple[int, int], fill: str = "#000000") -> None:
	 w, h = size
	 svg_paths = []
	 for p in paths:
		 segs = []
		 for i in range(len(p)):
			 p0 = p[i]
			 p1 = p[(i + 1) % len(p)]
			 mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
			 segs.append(QuadraticBezier(p0, mid, p1))
		 svg_paths.append(SVGPath(*segs))
	 wsvg(svg_paths, filename=str(out_file), dimensions=(w, h))
