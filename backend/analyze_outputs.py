import os
import argparse
from PIL import Image
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None

try:
    import pytesseract
except Exception:
    pytesseract = None

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "production", "tune")

def var_laplacian_gray(npimg):
    if cv2 is not None:
        g = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    # fallback gradient magnitude variance
    g = np.mean(npimg, axis=2).astype(np.float32)
    gy, gx = np.gradient(g)
    grad = np.sqrt(gx*gx + gy*gy)
    return float(grad.var())

def compute_ssim(a, b):
    if ssim is None:
        return None
    # convert to gray arrays
    ag = np.mean(a, axis=2).astype(np.float32)
    bg = np.mean(b, axis=2).astype(np.float32)
    try:
        return float(ssim(ag, bg))
    except Exception:
        return None

def ocr_check(img):
    if pytesseract is None:
        return None
    try:
        txt = pytesseract.image_to_string(img)
        return txt.strip()
    except Exception:
        return None

def analyze_file(path, ref_img=None):
    info = {}
    info["path"] = path
    info["filesize"] = os.path.getsize(path)
    with Image.open(path) as im:
        im.load()
        info["format"] = im.format
        info["mode"] = im.mode
        info["size"] = im.size
        arr = np.array(im.convert("RGB"))
    info["mean_color"] = tuple(map(float, arr.mean(axis=(0,1))))
    info["std_color"] = tuple(map(float, arr.std(axis=(0,1))))
    info["lap_var"] = var_laplacian_gray(arr)
    info["is_small"] = info["size"][0] < 256 or info["size"][1] < 256
    info["ocr"] = None
    # quick OCR on central area
    try:
        h,w = info["size"][1], info["size"][0]
        crop = Image.open(path).convert("RGB").crop((w//6, h//6, w*5//6, h*5//6))
        info["ocr"] = ocr_check(crop)
    except Exception:
        info["ocr"] = None
    info["ssim"] = None
    if ref_img is not None:
        info["ssim"] = compute_ssim(arr, ref_img)
    return info

def main(ref_path=None):
    files = sorted([os.path.join(OUT_DIR,f) for f in os.listdir(OUT_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
    report_lines = []
    ref_img = None
    if ref_path:
        with Image.open(ref_path) as r:
            ref_img = np.array(r.convert("RGB"))
    if not files:
        print("No images found in", OUT_DIR)
        return
    for f in files:
        print("Analyzing", f)
        info = analyze_file(f, ref_img)
        lines = [
            f"FILE: {info['path']}",
            f" size: {info['size']} bytes: {info['filesize']}",
            f" format/mode: {info['format']}/{info['mode']}",
            f" mean_color: {tuple(round(c,2) for c in info['mean_color'])} std: {tuple(round(c,2) for c in info['std_color'])}",
            f" laplacian_variance (sharpness proxy): {info['lap_var']:.2f}",
            f" small_image: {info['is_small']}",
            f" ocr_text_snippet: {repr(info['ocr'])}",
            f" ssim_vs_ref: {info['ssim']}",
            "-"*60,
        ]
        report_lines += lines
    out_report = os.path.join(OUT_DIR, "analysis_report.txt")
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("Wrote report to", out_report)
    print("\n".join(report_lines))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", help="optional reference image for SSIM comparison")
    args = p.parse_args()
    main(args.ref)