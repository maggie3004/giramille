from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import base64
import json
import os, inspect, traceback, re, time
from typing import List, Dict, Optional
import uuid
import threading
import logging
import random

# use relative import so backend.giramille_production is resolved when backend is a package
from .giramille_production import initialize_production_system

app = FastAPI()

# dev-friendly CORS: allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ensure storage dir exists and mount it so /storage/<file> is served
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

# history file path (persisted)
HISTORY_FILE = os.path.join(STORAGE_DIR, "history.json")
HISTORY_LIMIT = 200  # keep most recent N entries

logger = logging.getLogger("backend.server")

def _read_history() -> List[Dict]:
    try:
        if not os.path.exists(HISTORY_FILE):
            return []
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read history file: {e}")
        return []

def _write_history(entries: List[Dict]):
    try:
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp, HISTORY_FILE)
    except Exception as e:
        logger.error(f"Failed to write history file: {e}")

def add_history_entry(entry: Dict):
    entries = _read_history()
    # prepend newest first
    entries.insert(0, entry)
    # cap length
    if len(entries) > HISTORY_LIMIT:
        entries = entries[:HISTORY_LIMIT]
    _write_history(entries)

# initialize model once (try several call signatures)
try:
    g = initialize_production_system(device="cpu")
except TypeError:
    try:
        g = initialize_production_system("cpu")
    except TypeError:
        g = initialize_production_system()

# model readiness info (add after initialize_production_system block)
try:
    MODEL_READY = bool(g and callable(getattr(g, "generate_image", None)))
except Exception:
    MODEL_READY = False
logger.info(f"Production generator ready = {MODEL_READY}")

# jobs persistence + helpers
JOBS_FILE = os.path.join(STORAGE_DIR, "jobs.json")
JOBS_LOCK = threading.Lock()

def _read_jobs() -> dict:
    try:
        if not os.path.exists(JOBS_FILE):
            return {}
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"_read_jobs failed: {e}")
        return {}

def _write_jobs(jobs: dict):
    try:
        tmp = JOBS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        os.replace(tmp, JOBS_FILE)
    except Exception as e:
        logger.error(f"_write_jobs failed: {e}")

def _update_job(job_id: str, patch: dict):
    with JOBS_LOCK:
        jobs = _read_jobs()
        job = jobs.get(job_id, {})
        job.update(patch)
        jobs[job_id] = job
        _write_jobs(jobs)

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
    """
    Receives JSON { prompt, style?, quality?, seed? }.
    Calls the production generator and returns {"success":true,"image":"http://.../storage/xx.png","final_prompt":...}
    Saves file into storage so history works reliably.
    """
    try:
        body = await req.json()
        prompt = body.get("prompt", "") or ""
        style = body.get("style", None)
        quality = body.get("quality", "high")  # prefer high quality by default
        seed = body.get("seed", None)

        final_prompt = expand_prompt(prompt, auto_expand=True)

        # call generator (g) with enforced high quality and pass seed/style
        result = g.generate_image(final_prompt, negative_prompt=None, seed=seed, style=style, quality=quality)

        if not result.get("success"):
            return JSONResponse({"success": False, "error": result.get("error", "generation failed")}, status_code=500)

        img_payload = result.get("image")
        # if bytes -> save to storage and return stable URL
        if isinstance(img_payload, (bytes, bytearray)):
            filename = f"gen_{int(time.time()*1000)}_{result.get('seed',0)}.png"
            path = os.path.join(STORAGE_DIR, filename)
            with open(path, "wb") as f:
                f.write(img_payload)
            url = f"http://127.0.0.1:5000/storage/{filename}"
            entry = {
                "filename": filename,
                "url": url,
                "prompt": result.get("prompt") or prompt,
                "final_prompt": result.get("final_prompt") or final_prompt,
                "seed": result.get("seed"),
                "quality": result.get("quality") or quality,
                "timestamp": int(time.time())
            }
            add_history_entry(entry)
            resp = {"success": True, "image": url, "final_prompt": entry["final_prompt"], "seed": entry["seed"]}
            return JSONResponse(resp)

        # if generator returned a path or URL, normalize it
        if isinstance(img_payload, str):
            url = img_payload
            if url.startswith("/storage"):
                url = f"http://127.0.0.1:5000{url}"
            resp = {"success": True, "image": url, "final_prompt": result.get("final_prompt") or final_prompt, "seed": result.get("seed")}
            return JSONResponse(resp)

        return JSONResponse({"success": False, "error": "unexpected image payload"}, status_code=500)

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"success": False, "error": str(e), "trace": tb}, status_code=500)


@app.post("/api/retouch")
async def api_retouch(req: Request, background_tasks: BackgroundTasks):
    """
    Schedules a retouch job and returns immediately with a job id.
    Frontend should poll /api/job/{job_id} for status.
    """
    try:
        body = await req.json()
        prompt = body.get("prompt", "") or ""
        seed = body.get("seed", None)
        style = body.get("style", None)
        quality = body.get("quality", "high")

        job_id = uuid.uuid4().hex
        job_entry = {
            "id": job_id,
            "status": "queued",
            "prompt": prompt,
            "style": style,
            "quality": quality,
            "created_at": int(time.time())
        }
        with JOBS_LOCK:
            jobs = _read_jobs()
            jobs[job_id] = job_entry
            _write_jobs(jobs)

        # schedule background processing
        background_tasks.add_task(_process_job, job_id, prompt, style, quality, seed, True)

        return JSONResponse({"success": True, "job_id": job_id})
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"success": False, "error": str(e), "trace": tb}, status_code=500)

@app.get("/api/job/{job_id}")
async def api_get_job(job_id: str):
    jobs = _read_jobs()
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"success": False, "error": "job not found"}, status_code=404)
    return job

@app.get("/api/jobs")
async def api_list_jobs():
    return _read_jobs()

@app.get("/api/history")
async def api_history():
    """
    Return list of generated image metadata (most recent first).
    Each entry: { filename, url, prompt, final_prompt, seed, timestamp }
    """
    try:
        entries = _read_history()
        return entries
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

async def _process_job(job_id: str, prompt: str, style: Optional[str], quality: Optional[str], seed: Optional[int], retouch: bool=False):
    """
    Background job worker: generates image, saves to storage, updates job record and history.
    """
    try:
        _update_job(job_id, {"status": "running", "started_at": int(time.time())})
        final_prompt = expand_prompt(prompt, auto_expand=True)
        if retouch:
            suffix = "clean linework, color correction, smooth shading, preserve composition, no text, no watermark"
            final_prompt = f"{final_prompt}, {suffix}" if final_prompt else suffix

        # call generator (sync) - generate_image returns dict with "success" and "image" bytes or url
        result = g.generate_image(final_prompt, negative_prompt=None, seed=seed, style=style, quality=(quality or "high"))
        if not result.get("success"):
            _update_job(job_id, {"status": "failed", "error": result.get("error", "generation failed")})
            return

        img_payload = result.get("image")
        if isinstance(img_payload, (bytes, bytearray)):
            filename = f"job_{job_id}_{int(time.time()*1000)}_{result.get('seed',0)}.png"
            path = os.path.join(STORAGE_DIR, filename)
            with open(path, "wb") as f:
                f.write(img_payload)
            url = f"http://127.0.0.1:5000/storage/{filename}"
            entry = {
                "filename": filename,
                "url": url,
                "prompt": result.get("prompt") or prompt,
                "final_prompt": result.get("final_prompt") or final_prompt,
                "seed": result.get("seed"),
                "quality": result.get("quality") or quality,
                "timestamp": int(time.time())
            }
            add_history_entry(entry)
            _update_job(job_id, {"status": "finished", "image_url": url, "final_prompt": entry["final_prompt"], "seed": entry["seed"], "finished_at": int(time.time())})
            return

        if isinstance(img_payload, str):
            url = img_payload
            if url.startswith("/storage"):
                url = f"http://127.0.0.1:5000{url}"
            entry = {
                "filename": os.path.basename(url),
                "url": url,
                "prompt": result.get("prompt") or prompt,
                "final_prompt": result.get("final_prompt") or final_prompt,
                "seed": result.get("seed"),
                "quality": result.get("quality") or quality,
                "timestamp": int(time.time())
            }
            add_history_entry(entry)
            _update_job(job_id, {"status": "finished", "image_url": url, "final_prompt": entry["final_prompt"], "seed": entry["seed"], "finished_at": int(time.time())})
            return

        _update_job(job_id, {"status": "failed", "error": "unexpected image payload"})
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Job processing failed")
        _update_job(job_id, {"status": "failed", "error": str(e), "trace": tb})

@app.get("/api/model_status")
async def api_model_status():
    """
    Returns whether the generator/pipeline appears loaded and basic diagnostics.
    """
    try:
        if not g:
            return {"ready": False, "details": {"error": "generator object is None"}}
        details = {}
        details["has_generate_image"] = callable(getattr(g, "generate_image", None))
        # check for pipe components (if present)
        pipe = getattr(g, "pipe", None)
        details["pipe_present"] = pipe is not None
        if pipe is not None:
            try:
                details["components"] = {
                    "unet": hasattr(pipe, "unet"),
                    "text_encoder": hasattr(pipe, "text_encoder"),
                    "vae": hasattr(pipe, "vae"),
                    "safety_checker": hasattr(pipe, "safety_checker"),
                }
            except Exception as e:
                details["pipe_inspect_error"] = str(e)
        # quick heuristic: if any komponent safetensors missing, warn (user logs earlier showed missing files)
        return {"ready": details["has_generate_image"] and details.get("pipe_present", True), "details": details}
    except Exception as e:
        return JSONResponse({"ready": False, "error": str(e)}, status_code=500)

@app.post("/api/test_generate")
async def api_test_generate(req: Request):
    """
    Quick test generate endpoint: runs a small/fast generation to verify pipeline.
    Accepts JSON { prompt?: string, quality?: 'fast'|'balanced'|'high' }.
    Saves result to storage and returns URL + diagnostics (does a short run if quality='fast').
    """
    try:
        body = await req.json()
        prompt = (body.get("prompt") or "test image for pipeline").strip()
        quality = (body.get("quality") or "fast").lower()
        seed = int(body.get("seed")) if body.get("seed") is not None else random.randint(1, 2**31 - 1)

        # quick ready check
        if not g or not callable(getattr(g, "generate_image", None)):
            return JSONResponse({"success": False, "error": "generator not ready, check /api/model_status"}, status_code=500)

        # run generator with conservative params when 'fast' to avoid long runs
        try:
            result = g.generate_image(prompt, negative_prompt=None, seed=seed, style=None, quality=quality)
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("test_generate failed executing generate_image")
            return JSONResponse({"success": False, "error": str(e), "trace": tb}, status_code=500)

        if not result.get("success"):
            return JSONResponse({"success": False, "error": result.get("error", "generation failed"), "meta": result}, status_code=500)

        img_payload = result.get("image")
        if isinstance(img_payload, (bytes, bytearray)):
            filename = f"test_{int(time.time()*1000)}_{seed}.png"
            path = os.path.join(STORAGE_DIR, filename)
            with open(path, "wb") as f:
                f.write(img_payload)
            url = f"http://127.0.0.1:5000/storage/{filename}"
            entry = {
                "filename": filename,
                "url": url,
                "prompt": prompt,
                "final_prompt": result.get("final_prompt") or prompt,
                "seed": result.get("seed") or seed,
                "quality": result.get("quality") or quality,
                "timestamp": int(time.time())
            }
            add_history_entry(entry)
            return {"success": True, "image_url": url, "diagnostics": {"final_prompt": entry["final_prompt"], "seed": entry["seed"], "quality": entry["quality"]}}
        else:
            # string URL case
            url = img_payload
            if url.startswith("/storage"):
                url = f"http://127.0.0.1:5000{url}"
            return {"success": True, "image_url": url, "diagnostics": {"final_prompt": result.get("final_prompt"), "seed": result.get("seed")}}
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("api_test_generate failed")
        return JSONResponse({"success": False, "error": str(e), "trace": tb}, status_code=500)