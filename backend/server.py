from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import inspect
import traceback

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
        prompt = payload.get("prompt", "")
        width = int(payload.get("width", 512))
        height = int(payload.get("height", 512))
        quality = payload.get("quality", "balanced")

        # Build call based on generator signature
        sig = inspect.signature(g.generate_image)
        params = set(sig.parameters.keys())

        call_kwargs = {}
        # common prompt parameter names
        if "prompt" in params:
            call_kwargs["prompt"] = prompt
        elif "text" in params:
            call_kwargs["text"] = prompt
        elif "prompt_text" in params:
            call_kwargs["prompt_text"] = prompt

        # size / width/height handling
        if "width" in params and "height" in params:
            call_kwargs["width"] = width
            call_kwargs["height"] = height
        elif "size" in params:
            call_kwargs["size"] = (width, height)
        elif "resolution" in params:
            call_kwargs["resolution"] = (width, height)

        # quality / mode
        if "quality" in params:
            call_kwargs["quality"] = quality
        elif "mode" in params:
            call_kwargs["mode"] = quality

        # Attempt keyword call first, fall back to sensible positional attempts
        try:
            res = g.generate_image(**call_kwargs)
        except TypeError:
            try:
                # try common positional signature: prompt, width, height, quality
                res = g.generate_image(prompt, width, height, quality)
            except TypeError:
                try:
                    res = g.generate_image(prompt)
                except Exception as e:
                    raise

        if res.get("success") and res.get("image"):
            b64 = base64.b64encode(res["image"]).decode("ascii")
            return JSONResponse({"success": True, "image": f"data:image/png;base64,{b64}"})
        return JSONResponse({"success": False, "error": res.get("error", "generation failed")}, status_code=500)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse({"success": False, "error": str(e), "traceback": tb}, status_code=500)