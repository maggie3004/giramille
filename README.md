# 🎨 AI Image Generation + Vectorization Pipeline (MIT)

This repository provides a comprehensive offline pipeline for:
- Training a small diffusion model (U-Net) from scratch
- Generating images from noise or with optional reference conditioning
- Segmenting and converting raster images to layered vector graphics (SVG, AI, EPS)
- Computing quality metrics (SSIM, perceptual proxy, IoU/Chamfer, noise detector)
- **Advanced Freepik-like Editor** with modular editing capabilities
- Multi-view generation (Adobe Illustrator-style angle variations)
- Scene graph management with real-time editing
- Asset integration and harmonization

All components are implemented with permissive-licensed libraries and require no network access.

## 🚀 Quick Start

### For Complete Setup Instructions
**👉 [See the Complete Setup Guide](COMPLETE_SETUP_GUIDE.md) for detailed A-Z instructions**

### Basic Pipeline
Follow the original quickstart guide below for the core image generation and vectorization pipeline.

### Advanced Editor
For the new Freepik-like advanced editor with modular editing:

**Windows:**
```bash
start_advanced.bat
```

**Linux/Mac:**
```bash
./start_advanced.sh
```

This will start both the backend API server and frontend editor. Access the advanced editor at `http://localhost:3000/editor`.

## Quickstart

1) Create environment

```bash
# Using conda
conda env create -f environment.yml
conda activate offline-genvec

# Or using pip
pip install -r requirements.txt
```

2) Prepare dataset

```bash
python scripts/prepare_dataset.py \
  --data_dir path/to/local_dataset \
  --out_dir data/augmented \
  --val_ratio 0.1 --test_ratio 0.1 \
  --augmentations affine hue elastic cutmix
```

3) Train diffusion model

```bash
python train.py hydra.run.dir=outputs/train \
  trainer.batch_size=16 trainer.max_steps=100000 \
  trainer.gpus=2 trainer.mixed_precision=true
```

4) Generate images

```bash
python infer.py --checkpoint outputs/train/last.ckpt --out_dir outputs/samples --num_images 8
```

5) Vectorize images

```bash
python vectorize.py --input_dir outputs/samples --out_dir outputs/vectors \
  --max_anchors 300 --max_layers 20
```

6) Launch UI

```bash
python ui.py --host 127.0.0.1 --port 5000
```

## Project Structure

```
.
├── configs/
│   └── config.yaml
├── scripts/
│   ├── prepare_dataset.py
│   ├── dataset_manifest.py
│   ├── offline_guard.py
│   └── seg_classes_template.json
├── src/
│   ├── data/
│   │   ├── augment.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── diffusion_unet.py
│   │   └── segnet.py
│   ├── utils/
│   │   ├── checkpoint.py
│   │   ├── distributed.py
│   │   └── metrics.py
│   └── vector/
│       ├── curve_fit.py
│       ├── postprocess.py
│       └── export.py
├── frontend/                    # Next.js Advanced Editor
│   ├── app/
│   │   ├── page.tsx            # Main Stage 1 UI
│   │   └── editor/
│   │       └── page.tsx        # Advanced Editor
│   ├── components/
│   │   ├── SceneGraph.tsx      # Layer management
│   │   ├── CanvasEditor.tsx    # Interactive canvas
│   │   ├── AssetUploader.tsx   # Asset management
│   │   ├── MultiViewGenerator.tsx # Multi-view AI
│   │   └── History.tsx         # Generation history
│   └── public/static/          # UI assets
├── backend/                     # Flask API Server
│   ├── app.py                  # Main API server
│   ├── run.py                  # Server startup
│   └── requirements.txt        # Backend dependencies
├── train.py
├── infer.py
├── vectorize.py
├── ui.py                       # Original Flask UI
├── requirements.txt
├── environment.yml
├── start_advanced.bat          # Windows startup script
├── start_advanced.sh           # Linux/Mac startup script
└── ADVANCED_FEATURES.md        # Advanced features documentation
```

## 🎨 Advanced Features

### Freepik-like Modular Editing
- **Scene Graph Management**: Hierarchical layer organization with drag-and-drop
- **Real-time Editing**: Non-destructive editing with instant visual feedback
- **Asset Integration**: Seamless import and harmonization of custom assets
- **Transform Operations**: Move, scale, rotate, and manipulate elements

### Multi-View Generation
- **8 View Angles**: Front, back, left, right, top, bottom, 3/4, profile
- **Style Preservation**: Maintains artistic consistency across all angles
- **Adobe Illustrator-like**: Similar to AI's multi-view generation feature
- **Batch Processing**: Generate multiple views simultaneously

### Professional Export
- **Vector Output**: Clean, editable SVG files with layer separation
- **High Resolution**: PNG export up to 4K resolution
- **Print Ready**: PDF export with proper formatting
- **Format Support**: SVG, AI (PDF-based), EPS, PNG

### Technical Architecture
- **Frontend**: Next.js + React + TypeScript + Tailwind CSS
- **Backend**: Flask API with async processing
- **Canvas**: HTML5 Canvas with WebGL acceleration
- **Storage**: In-memory scene graphs (extensible to database)

## 📚 Documentation

- **[Complete Setup Guide](COMPLETE_SETUP_GUIDE.md)** - A-Z setup instructions for new users
- **[Advanced Features Guide](ADVANCED_FEATURES.md)** - Comprehensive documentation of all advanced features
- **[API Reference](ADVANCED_FEATURES.md#api-reference)** - Complete backend API documentation
- **[Usage Examples](ADVANCED_FEATURES.md#usage-guide)** - Step-by-step usage guide

## 🚀 Getting Started with Advanced Editor

1. **Start the servers:**
   ```bash
   # Windows
   start_advanced.bat
   
   # Linux/Mac
   ./start_advanced.sh
   ```

2. **Access the editor:**
   - Main UI: `http://localhost:3000`
   - Advanced Editor: `http://localhost:3000/editor`

3. **Basic workflow:**
   - Upload assets via drag-and-drop
   - Create scenes with multiple layers
   - Generate multi-view variations
   - Export in various formats

## Notes
- Everything runs offline. No downloads occur during runtime.
- The LPIPS metric is implemented as an offline perceptual proxy combining multi-scale SSIM and gradient magnitude similarity to avoid model downloads.
- AI export is produced via PDF-based workflow for compatibility.
- Advanced editor requires both frontend and backend servers running.

## License
MIT