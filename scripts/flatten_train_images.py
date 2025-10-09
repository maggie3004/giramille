from pathlib import Path
import shutil

# Source: All category subfolders
src_root = Path('data/train')
dest = src_root / 'images'
dest.mkdir(parents=True, exist_ok=True)
exts = {'.png', '.jpg', '.jpeg'}

count = 0
for sub in src_root.iterdir():
    if sub.is_dir() and sub.name != 'images' and sub.name != 'masks':
        for imgf in sub.rglob('*'):
            if imgf.suffix.lower() in exts:
                target = dest / imgf.name
                shutil.copyfile(imgf, target)
                count += 1
                print(f"Copied {imgf} -> {target}")
print(f"[DONE] Flattened/collated {count} images into data/train/images/")
