from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np

images_dir = Path('data/train/images')
masks_dir = Path('data/train/masks')
masks_dir.mkdir(parents=True, exist_ok=True)

exts = {'.png', '.jpg', '.jpeg'}
count = 0
skipped = []

for imgf in images_dir.glob('*'):
    if imgf.suffix.lower() in exts:
        try:
            img = Image.open(imgf)
            w, h = img.size
            arr = np.ones((h, w), dtype=np.uint8)
            out_mask = Image.fromarray(arr)
            out_path = masks_dir / imgf.name
            out_mask.save(out_path)
            count += 1
            print(f"Dummy mask created: {out_path}")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[SKIP] {imgf}: unreadable/corrupted. ({e})")
            skipped.append((str(imgf), 'corrupted'))
        except Exception as e:
            if 'DecompressionBombError' in str(type(e)) or 'DecompressionBomb' in str(e):
                print(f"[SKIP] {imgf}: too large / DecompressionBombError. ({e})")
                skipped.append((str(imgf), 'decompression bomb'))
            else:
                print(f"[SKIP] {imgf}: unknown error. ({e})")
                skipped.append((str(imgf), 'other error'))
print(f"[DONE] Created {count} dummy masks in data/train/masks/")
if skipped:
    print(f"[INFO] Skipped {len(skipped)} files:")
    for f, reason in skipped:
        print(f"   - {f}: {reason}")
