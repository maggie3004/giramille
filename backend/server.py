from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import inspect, traceback, base64, re

# use relative import so backend.giramille_production is resolved when backend is a package
from .giramille_production import initialize_production_system

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# initialize model once (try several call signatures)
try:
    g = initialize_production_system(device="cpu")
except TypeError:
    try:
        g = initialize_production_system("cpu")
    except TypeError:
        g = initialize_production_system()

# helper: expand short prompts with a safe template
def expand_prompt(prompt: str, auto_expand: bool = True) -> str:
    if not prompt or not auto_expand:
        return prompt
    p = prompt.strip()
    low = p.lower()
    # if user already included strong descriptors, don't append
    keywords = ("cartoon","vector","cel-shad","cel-shaded","outline","illustration","comic","flat","3/4","front 3/4")
    if any(k in low for k in keywords) and len(p.split()) >= 6:
        return p
    # avoid duplicating "no watermark"/"no text"
    suffix = "thick black outlines, cel-shaded, front 3/4 view, high contrast, flat background, simplified shapes, no text, no watermark"
    # if prompt ends with punctuation ensure single comma
    if re.search(r"[.!?]$", p):
        p = p[:-1]
    return p + ", " + suffix

@app.get("/", response_class=PlainTextResponse)
async def index():
    return "Giramille backend running"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/generate")
async def api_generate(req: Request):
    try:
        payload = await req.json()
        prompt = payload.get("prompt", "") or ""
        # frontend may pass auto_expand boolean; default True
        auto_expand = payload.get("auto_expand", True)
        # expand prompt if appropriate
        prompt = expand_prompt(prompt, auto_expand=auto_expand)

        negative_prompt = payload.get("negative_prompt")
        seed = payload.get("seed")
        style = payload.get("style")
        quality = payload.get("quality")

        call_kwargs = {}
        if prompt is not None:
            call_kwargs["prompt"] = prompt
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            call_kwargs["seed"] = int(seed)
        if style is not None:
            call_kwargs["style"] = style
        if quality is not None:
            call_kwargs["quality"] = quality

        res = g.generate_image(**call_kwargs)

        if res.get("success") and res.get("image"):
            b64 = base64.b64encode(res["image"]).decode("ascii")
            return JSONResponse({"success": True, "image": f"data:image/png;base64,{b64}", "meta": {k: res.get(k) for k in ('generation_time','prompt','style','quality') if k in res}})
        return JSONResponse({"success": False, "error": res.get("error","generation failed")}, status_code=500)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse({"success": False, "error": str(e), "traceback": tb}, status_code=500)