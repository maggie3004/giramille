# 🚀 Complete Setup Guide - AI Image Generation Giramille

This is a comprehensive A-Z guide for anyone who wants to clone and run this project from scratch.

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: At least 10GB free space
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)

### Required Software
1. **Python 3.8+** (Python 3.11 recommended)
2. **Node.js 16+** (Node.js 18+ recommended)
3. **Git** (for cloning the repository)
4. **CUDA Toolkit** (if using GPU for training)

---

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/maggie3004/giramille.git
cd giramille
```

### Step 2: Install Python Dependencies

#### Option A: Using Virtual Environment (Recommended)

**Windows:**
```bash
# Install Python 3.11 (if not already installed)
winget install -e --id Python.Python.3.11

# Close and reopen PowerShell, then:
py -3.11 -V  # Verify installation
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

#### Option B: Using Conda

```bash
# Create conda environment
conda env create -f environment.yml
conda activate offline-genvec

# Install additional dependencies
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 4: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

---

## 🚀 Running the Application

### Quick Start (Recommended)

**Windows:**
```bash
start_advanced.bat
```

**Linux/macOS:**
```bash
chmod +x start_advanced.sh
./start_advanced.sh
```

This will automatically:
- Install all dependencies
- Start the backend server (Flask API)
- Start the frontend server (Next.js)
- Open both servers in separate windows

### Manual Start (Alternative)

If you prefer to start servers manually:

**Terminal 1 - Backend:**
```bash
cd backend
python run.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Access the Application

Once both servers are running:
- **Main UI**: http://localhost:3000
- **Advanced Editor**: http://localhost:3000/editor
- **Backend API**: http://localhost:5000

---

## 🎯 Available Features

### 1. Basic Image Generation
- **Prompt-based generation**: Enter text prompts to generate images
- **Predefined objects**: 100+ objects from the Giramille dataset
- **Style options**: Different styles for PNG vs Vector output
- **Color recognition**: Automatic color extraction from prompts

### 2. Advanced Editor (Freepik-like)
- **Scene Graph Management**: Hierarchical layer organization
- **Real-time Editing**: Non-destructive editing with instant feedback
- **Asset Integration**: Upload and harmonize custom assets
- **Transform Operations**: Move, scale, rotate elements
- **Multi-view Generation**: 8 different viewing angles

### 3. Vectorization Pipeline
- **SVG Export**: Clean, editable vector files
- **AI Export**: Adobe Illustrator compatible files
- **EPS Export**: Print-ready vector format
- **High Resolution**: PNG export up to 4K

---

## 🏗️ Project Structure

```
giramille/
├── 📁 frontend/                 # Next.js React frontend
│   ├── 📁 app/                  # Main pages and components
│   │   ├── page.tsx            # Main Stage 1 UI
│   │   └── editor/             # Advanced Editor
│   ├── 📁 components/           # Reusable UI components
│   │   ├── SceneGraph.tsx      # Layer management
│   │   ├── CanvasEditor.tsx    # Interactive canvas
│   │   ├── AssetUploader.tsx   # Asset management
│   │   ├── MultiViewGenerator.tsx # Multi-view AI
│   │   └── History.tsx         # Generation history
│   └── 📁 public/static/        # UI assets and images
├── 📁 backend/                  # Flask API server
│   ├── app.py                  # Main API server
│   ├── run.py                  # Server startup
│   └── requirements.txt        # Backend dependencies
├── 📁 src/                      # Core Python modules
│   ├── 📁 models/              # Neural network architectures
│   ├── 📁 data/                # Dataset handling
│   ├── 📁 utils/               # Utilities and metrics
│   └── 📁 vector/              # Vectorization pipeline
├── 📁 scripts/                  # Automation scripts
├── 📁 configs/                  # Configuration files
├── 📁 models/                   # Pre-trained model files
├── 📁 static/                   # Static assets
├── 📁 exports/                  # Generated outputs
├── 📁 logs/                     # Log files
├── 📄 requirements.txt          # Main Python dependencies
├── 📄 environment.yml           # Conda environment file
├── 📄 start_advanced.bat        # Windows startup script
├── 📄 start_advanced.sh         # Linux/Mac startup script
└── 📄 README.md                 # This file
```

---

## 🎨 Dataset Information

The project includes a curated dataset with 321 images organized into categories:

### Categories Available:
- **Animals**: bird, cat, dog, fish, butterfly, horse, bear, chick, ant, frog, crocodile, moose, t-rex, fairy, witch
- **Objects**: car, airplane, train, bus, boat, house, castle, tree, flower, hat, star, heart, sun, moon, cloud, ball, book, cup
- **Scenes**: forest, beach, mountain, city, school, farm, prison, stage, park, bridge, statue, sky, ground, wood, water, rail
- **Food**: apple, bread, milk, banana, ice cream, fish food
- **Characters**: giramille, indian, firefighter, chef
- **Items**: wand, fishing rod, surfboard, mask, flag, map, leaf, rainbow, clothespin, belt, tutu, bow, frame, sign
- **Colors**: red, blue, green, yellow, purple, pink, brown, black, white, orange
- **Holidays**: christmas, easter, birthday, congratulations
- **Hygiene Products**: shampoo, soap, toothbrush, mouthwash, dental floss, diaper, diaper cream, conditioner, wet wipes, hand sanitizer

---

## 🔧 Advanced Usage

### Training Custom Models

```bash
# Prepare dataset
python scripts/prepare_dataset.py --data-dir data/train --out-dir data/processed

# Start training
python train.py --config configs/training_config.yaml

# Generate images
python infer.py --checkpoint checkpoints/best_model.pth --prompt "a red bird"

# Vectorize images
python vectorize.py --input outputs/generated --output outputs/vectors
```

### Using the Original Flask UI

```bash
python ui.py --host 127.0.0.1 --port 5000
```

### API Endpoints

The backend provides several API endpoints:

- `POST /api/generate` - Generate images from prompts
- `POST /api/vectorize` - Convert images to vectors
- `GET /api/history` - Get generation history
- `POST /api/upload` - Upload custom assets

---

## 🐛 Troubleshooting

### Common Issues

**1. Python not found:**
```bash
# Windows
winget install -e --id Python.Python.3.11

# Linux
sudo apt update
sudo apt install python3.11 python3.11-venv

# macOS
brew install python@3.11
```

**2. Node.js not found:**
```bash
# Windows
winget install -e --id OpenJS.NodeJS

# Linux
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS
brew install node
```

**3. CUDA not available:**
- Install CUDA Toolkit from NVIDIA website
- Verify with: `nvidia-smi`
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

**4. Port already in use:**
```bash
# Kill processes using ports 3000 or 5000
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/macOS
lsof -ti:3000 | xargs kill -9
```

**5. Permission denied (Linux/macOS):**
```bash
chmod +x start_advanced.sh
```

### Getting Help

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Verify all dependencies are installed correctly
3. Ensure ports 3000 and 5000 are available
4. Check that Python and Node.js are in your PATH

---

## 📚 Additional Documentation

- **[Advanced Features Guide](ADVANCED_FEATURES.md)** - Comprehensive documentation of all advanced features
- **[API Reference](ADVANCED_FEATURES.md#api-reference)** - Complete backend API documentation
- **[Usage Examples](ADVANCED_FEATURES.md#usage-guide)** - Step-by-step usage guide
- **[Setup Complete](SETUP_COMPLETE.md)** - Detailed setup status and features

---

## 🎉 You're Ready!

Once you've completed these steps, you should have:
- ✅ A fully functional AI image generation system
- ✅ Advanced Freepik-like editing capabilities
- ✅ Multi-view generation features
- ✅ Professional vector export options
- ✅ Complete offline operation

**Happy creating! 🎨**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📞 Support

For support, please open an issue on GitHub or contact the development team.
