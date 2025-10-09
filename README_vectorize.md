# Vectorization Workflow (Backend Only)

## 1. Prepare Data
- Organize input images and segmentation masks:
  - `data/seg/images/`  — input PNG/JPEGs
  - `data/seg/masks/`   — grayscale PNGs (pixel=class index, e.g. 0 for background, 1 for object1, etc.)
- Each mask must align (by filename) with its corresponding image.

## 2. Train Segmentation Model
Run:
```
python scripts/train_segnet.py
```
- Change paths in script if needed (should match `data/seg/images` and `data/seg/masks`).
- Checkpoints will be saved as `smallunet_epochXX.pth`, final weights as `smallunet_best.pth`.

## 3. Vectorize Images
Run:
```
python vectorize.py --input-dir <folder_with_pngs_to_vectorize> --weights smallunet_best.pth
```
- SVG/EPS/PDF vectors will be written to `outputs/vectors/`.
- Each class forms a separate vector layer (editable in Illustrator/Corel).

## 4. Inspect Output
- Open vectors in a vector editor. Each prominent region/class should be a separate layer for editing.
- For best results, segmentation masks should have accurate and clean class boundaries.

## Troubleshooting
- If `vectorize.py` exits with missing or wrong weights: ensure you provide correct `--weights` argument.
- If output is messy, retrain or add more varied mask data for your objects/classes.
