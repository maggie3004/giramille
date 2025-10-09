import sys
import os
import base64
from pathlib import Path
import subprocess
import glob

# Auto-discover data/test, data/val, or data/giramille_processed/images
DATA_ROOT = Path("data/")
candidates = []
for sub in ["test", "val", "giramille_processed", "train"]:
    imgdir = DATA_ROOT / sub / "images"
    if imgdir.exists():
        for f in imgdir.glob("*.png"):
            candidates.append(f)
if len(candidates) < 3:
    print("[ERROR] Need at least 3 PNGs in (data/test|val|giramille_processed/images) for contract POC export.")
    exit(1)

export_dir = Path("outputs/POC_pocimages/")
export_dir.mkdir(parents=True, exist_ok=True)

# Copy exactly 3 distinct images (character, animal/object, scenario for demo only)
for i, imgf in enumerate(candidates[:3]):
    dest = export_dir / f"poc{i+1}_{imgf.name}"
    with open(imgf, "rb") as fin, open(dest, "wb") as fout:
        fout.write(fin.read())
    print(f"[POC] Copied sample image: {imgf} => {dest}")

# Vectorize POC images
vectorized = Path("outputs/POC_vectors/")
vectorized.mkdir(parents=True, exist_ok=True)
weights = Path("smallunet_best.pth")
cmd = [
    "python", "vectorize.py",
    "--input-dir", str(export_dir),
    "--weights", str(weights)
]
print(f"[POC] Vectorizing contract proof images...")
subprocess.run(cmd, check=True)
print("[DONE] POC images vectorized. Open outputs/POC_vectors/ for contract validation.")
