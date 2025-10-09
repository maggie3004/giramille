from pathlib import Path
import shutil

# Source: All category subfolders
src_root = Path('data/train')
dest = src_root / 'masks'
dest.mkdir(parents=True, exist_ok=True)
exts = {'.png', '.jpg', '.jpeg'}

count = 0
for sub in src_root.iterdir():
    if sub.is_dir() and sub.name != 'images' and sub.name != 'masks':
        # For each mask in subfolder called masks/ (usually next to images)
        mask_dir = sub / 'masks'
        if mask_dir.exists():
            for maskf in mask_dir.rglob('*'):
                if maskf.suffix.lower() in exts:
                    target = dest / maskf.name
                    shutil.copyfile(maskf, target)
                    count += 1
                    print(f"Copied {maskf} -> {target}")
print(f"[DONE] Flattened/collated {count} masks into data/train/masks/")
