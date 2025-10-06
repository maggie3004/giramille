<<<<<<< HEAD
﻿from pathlib import Path
from typing import Optional
import io
import glob

from flask import Flask, request, send_file, jsonify, render_template_string
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np

from src.models.diffusion_unet import UNetModel
from src.utils.checkpoint import load_checkpoint
from src.vector.curve_fit import contours_to_beziers, beziers_to_svg
from src.vector.postprocess import reduce_anchors, merge_layers
from src.models.segnet import SmallUNet

app = Flask(__name__, static_folder="static", static_url_path="/static")

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Offline Gen+Vector</title>
  <style>
    body { margin: 0; background: #0b1b3a; overflow-x: hidden; }
    .canvas-wrap { position: relative; width: 1365px; margin: 0 auto; }
    .canvas { width: 1365px; height: 900px; background: center/cover no-repeat url('/static/1. Tela principal.png'); }
    /* Clickable hitboxes (absolute). Coordinates tuned to the provided mock */
    .hit { position: absolute; cursor: pointer; }
    /* Buttons */
    #hit-gen-vector { left: 490px; top: 212px; width: 380px; height: 98px; }
    #hit-gen-png { left: 490px; top: 335px; width: 380px; height: 98px; }
    #hit-retocar { left: 120px; top: 730px; width: 260px; height: 100px; }
    #hit-resize { left: 410px; top: 730px; width: 260px; height: 100px; }
    #hit-cancelar { left: 700px; top: 730px; width: 260px; height: 100px; }
    #hit-exportar { left: 990px; top: 730px; width: 260px; height: 100px; }
    /* File picker over the history panel area */
    #fileInput { position: absolute; left: 1010px; top: 115px; width: 280px; height: 430px; opacity: 0.001; }
    /* Preview area inside the big frame */
    #preview { position: absolute; left: 260px; top: 470px; width: 840px; height: 380px; object-fit: contain; border-radius: 10px; }
    /* Hidden controls retained for functionality */
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="canvas-wrap">
    <div class="canvas"></div>
    <input id="fileInput" type="file" accept="image/*" />
    <img id="preview" />
    <div id="hit-gen-vector" class="hit" title="Gerar Vetor / Generate Vector"></div>
    <div id="hit-gen-png" class="hit" title="Gerar PNG / Generate PNG"></div>
    <div id="hit-retocar" class="hit" title="Retocar / Retouch"></div>
    <div id="hit-resize" class="hit" title="Redimensionar / Resize"></div>
    <div id="hit-cancelar" class="hit" title="Cancelar / Cancel"></div>
    <div id="hit-exportar" class="hit" title="Exportar / Export"></div>

    <!-- Hidden inputs for config -->
    <input id="image_size" class="hidden" type="number" value="256" />
    <input id="steps" class="hidden" type="number" value="50" />
    <input id="channels" class="hidden" type="number" value="64" />
    <input id="checkpoint" class="hidden" type="text" value="outputs/checkpoints/last.pt" />
    <input id="max_layers" class="hidden" type="number" value="20" />
    <input id="max_anchors" class="hidden" type="number" value="300" />
    <input id="vec_image_size" class="hidden" type="number" value="256" />
  </div>

  <script>
    const hitGenVector = document.getElementById('hit-gen-vector');
    const hitGenPNG = document.getElementById('hit-gen-png');
    const hitExportar = document.getElementById('hit-exportar');
    const hitResize = document.getElementById('hit-resize');
    const hitRetocar = document.getElementById('hit-retocar');
    const hitCancelar = document.getElementById('hit-cancelar');

    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');

    const imageSize = document.getElementById('image_size');
    const steps = document.getElementById('steps');
    const channels = document.getElementById('channels');
    const checkpoint = document.getElementById('checkpoint');

    const vecImageSize = document.getElementById('vec_image_size');
    const maxLayers = document.getElementById('max_layers');
    const maxAnchors = document.getElementById('max_anchors');

    let lastSVG = null;

    hitGenPNG.addEventListener('click', async () => {
      const data = new FormData();
      data.append('image_size', imageSize.value);
      data.append('steps', steps.value);
      data.append('channels', channels.value);
      data.append('checkpoint', checkpoint.value);
      const res = await fetch('/generate', { method: 'POST', body: data });
      const blob = await res.blob();
      preview.src = URL.createObjectURL(blob);
      lastSVG = null;
    });

    hitGenVector.addEventListener('click', async () => {
      if (!fileInput.files || fileInput.files.length === 0) {
        alert('Selecione uma imagem / Select an image');
        return;
      }
      const data = new FormData();
      data.append('image', fileInput.files[0]);
      data.append('max_layers', maxLayers.value);
      data.append('max_anchors', maxAnchors.value);
      data.append('image_size', vecImageSize.value);
      const res = await fetch('/vectorize', { method: 'POST', body: data });
      const out = await res.json();
      lastSVG = out.svg_path;
      alert('SVG salvo em / saved at: ' + out.svg_path);
    });

    hitExportar.addEventListener('click', async () => {
      const url = lastSVG ? `/export_last?hint=${encodeURIComponent(lastSVG)}` : '/export_last';
      const res = await fetch(url);
      if (res.status !== 200) { alert('Nada para exportar / Nothing to export'); return; }
      const blob = await res.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'vector_export.svg';
      document.body.appendChild(a);
      a.click();
      a.remove();
    });

    hitResize.addEventListener('click', () => {
      const v = prompt('Novo tamanho / New size (ex: 256):', imageSize.value);
      if (v) { imageSize.value = v; vecImageSize.value = v; }
    });

    hitRetocar.addEventListener('click', () => {
      alert('Retocar / Retouch (em breve / coming soon)');
    });

    hitCancelar.addEventListener('click', () => {
      fileInput.value = '';
      preview.src = '';
      lastSVG = null;
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
	 return render_template_string(INDEX_HTML)


@app.route("/health")
def health():
	 return jsonify({"ok": True})


@app.route("/generate", methods=["POST"])
def generate():
	 image_size = int(request.form.get("image_size", 256))
	 steps = int(request.form.get("steps", 50))
	 channels = int(request.form.get("channels", 64))
	 checkpoint = request.form.get("checkpoint", "outputs/checkpoints/last.pt")

	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	 model = UNetModel(in_channels=3, base_channels=channels).to(device)
	 if Path(checkpoint).exists():
		 state = load_checkpoint(Path(checkpoint), map_location=device)
		 model.load_state_dict(state["model"])
	 model.eval()

	 timesteps = 1000
	 betas = torch.linspace(1e-4, 2e-2, timesteps, device=device)
	 alphas = 1.0 - betas
	 alphas_cumprod = torch.cumprod(alphas, dim=0)
	 alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
	 posterior_var = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

	 def p_sample(x, t_idx):
		 with torch.no_grad():
			 pred_noise = model(x, torch.full((x.size(0),), t_idx, device=device, dtype=torch.long))
			 alpha_t = alphas[t_idx]
			 alpha_bar_t = alphas_cumprod[t_idx]
			 sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
			 sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
			 dir_xt = torch.sqrt(1 - alpha_t) * pred_noise
			 noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
			 var = torch.sqrt(posterior_var[t_idx]) * noise
			 x = sqrt_recip_alpha * (x - dir_xt) + var
			 return x

	 x = torch.randn(1, 3, image_size, image_size, device=device)
	 t_list = torch.linspace(timesteps - 1, 0, steps, dtype=torch.long)
	 for t_idx in t_list:
		 x = p_sample(x, int(t_idx))
	 x = (x.clamp(-1, 1) + 1) / 2.0
	 img = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]

	 _, buf = cv2.imencode(".png", img)
	 return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png", as_attachment=False, download_name="sample.png")


@app.route("/vectorize", methods=["POST"])
def vectorize():
	 max_layers = int(request.form.get("max_layers", 20))
	 max_anchors = int(request.form.get("max_anchors", 300))
	 image_size = int(request.form.get("image_size", 256))

	 file = request.files.get("image")
	 if file is None:
		 return jsonify({"error": "image required"}), 400
	 data = np.frombuffer(file.read(), np.uint8)
	 img = cv2.imdecode(data, cv2.IMREAD_COLOR)
	 if img is None:
		 return jsonify({"error": "invalid image"}), 400
	 h0, w0 = img.shape[:2]
	 img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
	 x = torch.from_numpy(img_resized[:, :, ::-1]).float().permute(2, 0, 1) / 255.0
	 x = x.unsqueeze(0)

	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	 model = SmallUNet(in_ch=3, num_classes=16).to(device)
	 model.eval()
	 with torch.no_grad():
		 logits = model(x.to(device))
		 mask = logits.argmax(dim=1)[0].byte().cpu().numpy()

	 layer_paths = []
	 for cls in range(16):
		 cls_mask = (mask == cls).astype(np.uint8) * 255
		 if cls_mask.sum() < 10:
			 continue
		 paths = contours_to_beziers(cls_mask, epsilon=1.5, max_segments=8)
		 paths = reduce_anchors(paths, max_anchors=max_anchors)
		 layer_paths.append(paths)

	 merged = merge_layers(layer_paths, max_layers=max_layers)
	 out_svg_dir = Path("outputs/ui")
	 out_svg_dir.mkdir(parents=True, exist_ok=True)
	 svg_path = out_svg_dir / "vector.svg"
	 beziers_to_svg(merged, svg_path, size=(w0, h0))

	 return jsonify({"svg_path": str(svg_path)})


@app.route('/export_last')
def export_last():
	 paths = []
	 for folder in ["outputs/ui", "outputs/vectors"]:
		 paths.extend([Path(p) for p in glob.glob(str(Path(folder) / "*.svg"))])
	 if not paths:
		 return ("No vector found", 404)
	 latest = max(paths, key=lambda p: p.stat().st_mtime)
	 return send_file(str(latest), mimetype="image/svg+xml", as_attachment=True, download_name=latest.name)


if __name__ == "__main__":
	 app.run(host="127.0.0.1", port=5000)
=======
﻿from pathlib import Path
from typing import Optional
import io
import glob

from flask import Flask, request, send_file, jsonify, render_template_string
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np

from src.models.diffusion_unet import UNetModel
from src.utils.checkpoint import load_checkpoint
from src.vector.curve_fit import contours_to_beziers, beziers_to_svg
from src.vector.postprocess import reduce_anchors, merge_layers
from src.models.segnet import SmallUNet

app = Flask(__name__, static_folder="static", static_url_path="/static")

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Offline Gen+Vector</title>
  <style>
    body { margin: 0; background: #0b1b3a; overflow-x: hidden; }
    .canvas-wrap { position: relative; width: 1365px; margin: 0 auto; }
    .canvas { width: 1365px; height: 900px; background: center/cover no-repeat url('/static/1. Tela principal.png'); }
    /* Clickable hitboxes (absolute). Coordinates tuned to the provided mock */
    .hit { position: absolute; cursor: pointer; }
    /* Buttons */
    #hit-gen-vector { left: 490px; top: 212px; width: 380px; height: 98px; }
    #hit-gen-png { left: 490px; top: 335px; width: 380px; height: 98px; }
    #hit-retocar { left: 120px; top: 730px; width: 260px; height: 100px; }
    #hit-resize { left: 410px; top: 730px; width: 260px; height: 100px; }
    #hit-cancelar { left: 700px; top: 730px; width: 260px; height: 100px; }
    #hit-exportar { left: 990px; top: 730px; width: 260px; height: 100px; }
    /* File picker over the history panel area */
    #fileInput { position: absolute; left: 1010px; top: 115px; width: 280px; height: 430px; opacity: 0.001; }
    /* Preview area inside the big frame */
    #preview { position: absolute; left: 260px; top: 470px; width: 840px; height: 380px; object-fit: contain; border-radius: 10px; }
    /* Hidden controls retained for functionality */
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="canvas-wrap">
    <div class="canvas"></div>
    <input id="fileInput" type="file" accept="image/*" />
    <img id="preview" />
    <div id="hit-gen-vector" class="hit" title="Gerar Vetor / Generate Vector"></div>
    <div id="hit-gen-png" class="hit" title="Gerar PNG / Generate PNG"></div>
    <div id="hit-retocar" class="hit" title="Retocar / Retouch"></div>
    <div id="hit-resize" class="hit" title="Redimensionar / Resize"></div>
    <div id="hit-cancelar" class="hit" title="Cancelar / Cancel"></div>
    <div id="hit-exportar" class="hit" title="Exportar / Export"></div>

    <!-- Hidden inputs for config -->
    <input id="image_size" class="hidden" type="number" value="256" />
    <input id="steps" class="hidden" type="number" value="50" />
    <input id="channels" class="hidden" type="number" value="64" />
    <input id="checkpoint" class="hidden" type="text" value="outputs/checkpoints/last.pt" />
    <input id="max_layers" class="hidden" type="number" value="20" />
    <input id="max_anchors" class="hidden" type="number" value="300" />
    <input id="vec_image_size" class="hidden" type="number" value="256" />
  </div>

  <script>
    const hitGenVector = document.getElementById('hit-gen-vector');
    const hitGenPNG = document.getElementById('hit-gen-png');
    const hitExportar = document.getElementById('hit-exportar');
    const hitResize = document.getElementById('hit-resize');
    const hitRetocar = document.getElementById('hit-retocar');
    const hitCancelar = document.getElementById('hit-cancelar');

    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');

    const imageSize = document.getElementById('image_size');
    const steps = document.getElementById('steps');
    const channels = document.getElementById('channels');
    const checkpoint = document.getElementById('checkpoint');

    const vecImageSize = document.getElementById('vec_image_size');
    const maxLayers = document.getElementById('max_layers');
    const maxAnchors = document.getElementById('max_anchors');

    let lastSVG = null;

    hitGenPNG.addEventListener('click', async () => {
      const data = new FormData();
      data.append('image_size', imageSize.value);
      data.append('steps', steps.value);
      data.append('channels', channels.value);
      data.append('checkpoint', checkpoint.value);
      const res = await fetch('/generate', { method: 'POST', body: data });
      const blob = await res.blob();
      preview.src = URL.createObjectURL(blob);
      lastSVG = null;
    });

    hitGenVector.addEventListener('click', async () => {
      if (!fileInput.files || fileInput.files.length === 0) {
        alert('Selecione uma imagem / Select an image');
        return;
      }
      const data = new FormData();
      data.append('image', fileInput.files[0]);
      data.append('max_layers', maxLayers.value);
      data.append('max_anchors', maxAnchors.value);
      data.append('image_size', vecImageSize.value);
      const res = await fetch('/vectorize', { method: 'POST', body: data });
      const out = await res.json();
      lastSVG = out.svg_path;
      alert('SVG salvo em / saved at: ' + out.svg_path);
    });

    hitExportar.addEventListener('click', async () => {
      const url = lastSVG ? `/export_last?hint=${encodeURIComponent(lastSVG)}` : '/export_last';
      const res = await fetch(url);
      if (res.status !== 200) { alert('Nada para exportar / Nothing to export'); return; }
      const blob = await res.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'vector_export.svg';
      document.body.appendChild(a);
      a.click();
      a.remove();
    });

    hitResize.addEventListener('click', () => {
      const v = prompt('Novo tamanho / New size (ex: 256):', imageSize.value);
      if (v) { imageSize.value = v; vecImageSize.value = v; }
    });

    hitRetocar.addEventListener('click', () => {
      alert('Retocar / Retouch (em breve / coming soon)');
    });

    hitCancelar.addEventListener('click', () => {
      fileInput.value = '';
      preview.src = '';
      lastSVG = null;
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
	 return render_template_string(INDEX_HTML)


@app.route("/health")
def health():
	 return jsonify({"ok": True})


@app.route("/generate", methods=["POST"])
def generate():
	 image_size = int(request.form.get("image_size", 256))
	 steps = int(request.form.get("steps", 50))
	 channels = int(request.form.get("channels", 64))
	 checkpoint = request.form.get("checkpoint", "outputs/checkpoints/last.pt")

	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	 model = UNetModel(in_channels=3, base_channels=channels).to(device)
	 if Path(checkpoint).exists():
		 state = load_checkpoint(Path(checkpoint), map_location=device)
		 model.load_state_dict(state["model"])
	 model.eval()

	 timesteps = 1000
	 betas = torch.linspace(1e-4, 2e-2, timesteps, device=device)
	 alphas = 1.0 - betas
	 alphas_cumprod = torch.cumprod(alphas, dim=0)
	 alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
	 posterior_var = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

	 def p_sample(x, t_idx):
		 with torch.no_grad():
			 pred_noise = model(x, torch.full((x.size(0),), t_idx, device=device, dtype=torch.long))
			 alpha_t = alphas[t_idx]
			 alpha_bar_t = alphas_cumprod[t_idx]
			 sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
			 sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
			 dir_xt = torch.sqrt(1 - alpha_t) * pred_noise
			 noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
			 var = torch.sqrt(posterior_var[t_idx]) * noise
			 x = sqrt_recip_alpha * (x - dir_xt) + var
			 return x

	 x = torch.randn(1, 3, image_size, image_size, device=device)
	 t_list = torch.linspace(timesteps - 1, 0, steps, dtype=torch.long)
	 for t_idx in t_list:
		 x = p_sample(x, int(t_idx))
	 x = (x.clamp(-1, 1) + 1) / 2.0
	 img = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]

	 _, buf = cv2.imencode(".png", img)
	 return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png", as_attachment=False, download_name="sample.png")


@app.route("/vectorize", methods=["POST"])
def vectorize():
	 max_layers = int(request.form.get("max_layers", 20))
	 max_anchors = int(request.form.get("max_anchors", 300))
	 image_size = int(request.form.get("image_size", 256))

	 file = request.files.get("image")
	 if file is None:
		 return jsonify({"error": "image required"}), 400
	 data = np.frombuffer(file.read(), np.uint8)
	 img = cv2.imdecode(data, cv2.IMREAD_COLOR)
	 if img is None:
		 return jsonify({"error": "invalid image"}), 400
	 h0, w0 = img.shape[:2]
	 img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
	 x = torch.from_numpy(img_resized[:, :, ::-1]).float().permute(2, 0, 1) / 255.0
	 x = x.unsqueeze(0)

	 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	 model = SmallUNet(in_ch=3, num_classes=16).to(device)
	 model.eval()
	 with torch.no_grad():
		 logits = model(x.to(device))
		 mask = logits.argmax(dim=1)[0].byte().cpu().numpy()

	 layer_paths = []
	 for cls in range(16):
		 cls_mask = (mask == cls).astype(np.uint8) * 255
		 if cls_mask.sum() < 10:
			 continue
		 paths = contours_to_beziers(cls_mask, epsilon=1.5, max_segments=8)
		 paths = reduce_anchors(paths, max_anchors=max_anchors)
		 layer_paths.append(paths)

	 merged = merge_layers(layer_paths, max_layers=max_layers)
	 out_svg_dir = Path("outputs/ui")
	 out_svg_dir.mkdir(parents=True, exist_ok=True)
	 svg_path = out_svg_dir / "vector.svg"
	 beziers_to_svg(merged, svg_path, size=(w0, h0))

	 return jsonify({"svg_path": str(svg_path)})


@app.route('/export_last')
def export_last():
	 paths = []
	 for folder in ["outputs/ui", "outputs/vectors"]:
		 paths.extend([Path(p) for p in glob.glob(str(Path(folder) / "*.svg"))])
	 if not paths:
		 return ("No vector found", 404)
	 latest = max(paths, key=lambda p: p.stat().st_mtime)
	 return send_file(str(latest), mimetype="image/svg+xml", as_attachment=True, download_name=latest.name)


if __name__ == "__main__":
	 app.run(host="127.0.0.1", port=5000)
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
