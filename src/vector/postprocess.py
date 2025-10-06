from typing import List, Tuple


def reduce_anchors(paths: List[List[Tuple[float, float]]], max_anchors: int) -> List[List[Tuple[float, float]]]:
	 out = []
	 for p in paths:
		 if len(p) <= max_anchors:
			 out.append(p)
		 else:
			 step = max(1, len(p) // max_anchors)
			 out.append(p[::step])
	 return out


def merge_layers(layer_paths: List[List[List[Tuple[float, float]]]], max_layers: int) -> List[List[Tuple[float, float]]]:
	 # Flatten then cap to max_layers by taking largest first
	 flat = [seg for layer in layer_paths for seg in layer]
	 if len(flat) <= max_layers:
		 return flat
	 sizes = [(i, len(p)) for i, p in enumerate(flat)]
	 sizes.sort(key=lambda x: x[1], reverse=True)
	 idx_keep = set(i for i, _ in sizes[:max_layers])
	 return [p for i, p in enumerate(flat) if i in idx_keep]
