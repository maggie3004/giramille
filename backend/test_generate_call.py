import inspect, os, traceback
from PIL import Image, ImageFilter
import io

# robust import (works whether you run the script from repo root or as module)
try:
    from giramille_production import initialize_production_system
except Exception:
    from backend.giramille_production import initialize_production_system

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs", "production", "tune")
os.makedirs(OUTDIR, exist_ok=True)

PROMPT = "A bold colorful 2D cartoon train, thick black outlines, cel-shaded, front 3/4 view, high contrast, flat background, simplified shapes, no text, no watermark"

def save_bytes(b, path):
    with open(path, "wb") as f:
        f.write(b)

def upscale_with_pillow(in_bytes, factor=2, out_path=None):
    im = Image.open(io.BytesIO(in_bytes))
    new_size = (im.width * factor, im.height * factor)
    im_up = im.resize(new_size, resample=Image.LANCZOS)
    # optional sharpen
    im_up = im_up.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    if out_path:
        im_up.save(out_path)
    return im_up

try:
    g = initialize_production_system()  # adapt if needed
    sig = inspect.signature(g.generate_image)
    supported = set(sig.parameters.keys())
    print("generate_image signature:", sig)

    tries = [
        {"name":"base_default","kwargs":{}},
        {"name":"high_quality_1024","kwargs":{"quality":"high","width":1024,"height":1024}},
        {"name":"best_1536","kwargs":{"quality":"best","width":1536,"height":1536}},
        {"name":"cartoon_style_1024","kwargs":{"style":"cartoon","quality":"high","width":1024,"height":1024}},
    ]

    for t in tries:
        name = t["name"]
        kw = t["kwargs"].copy()
        # always pass prompt (map to supported param name)
        if "prompt" in supported:
            kw["prompt"] = PROMPT
            call_kwargs = {k: v for k, v in kw.items() if k in supported}
        elif "text" in supported:
            kw["text"] = PROMPT
            call_kwargs = {k: v for k, v in kw.items() if k in supported}
        else:
            # no keyword prompt supported — call positionally
            call_kwargs = None

        print("Trying:", name, call_kwargs if call_kwargs is not None else "(positional prompt)")

        try:
            if call_kwargs is not None:
                res = g.generate_image(**call_kwargs)
            else:
                res = g.generate_image(PROMPT)
        except Exception as e:
            print(f"Attempt '{name}' exception:", e)
            traceback.print_exc()
            continue

        if not isinstance(res, dict):
            print("unexpected response type:", type(res))
            continue

        print("keys:", list(res.keys()))
        if res.get("success") and res.get("image"):
            img_bytes = res["image"]
            if not isinstance(img_bytes, (bytes, bytearray)):
                print("image is not raw bytes, attempting to decode if base64 string")
                try:
                    import base64
                    img_bytes = base64.b64decode(img_bytes.split(",", 1)[1] if "," in img_bytes else img_bytes)
                except Exception:
                    print("failed to decode image bytes")
                    continue

            out_path = os.path.join(OUTDIR, f"{name}.png")
            save_bytes(img_bytes, out_path)
            up_path = os.path.join(OUTDIR, f"{name}_up.png")
            upscale_with_pillow(img_bytes, factor=2, out_path=up_path)
            print("saved:", out_path, "upscaled:", up_path)
        else:
            print("failed:", res.get("error"))
except Exception:
    traceback.print_exc()