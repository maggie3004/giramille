import sys
import os
from pathlib import Path
# Insert the project root so src/ is always importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError, ImageFile
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from models.segnet import SmallUNet
import warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Folder Discovery ---
DATA_ROOT = Path("data/")
dataset_choices = []
candidates_printed = False
for sub in ["train", "val", "giramille_processed", "processed"]:
    base = DATA_ROOT / sub
    img = base / "images"
    mask = base / "masks"
    print(f"[DEBUG] Checking: {img} and {mask}")
    if img.exists() and mask.exists():
        files_img = list(img.glob("*"))
        files_mask = list(mask.glob("*"))
        print(f"  [FOUND] {len(files_img)} images, {len(files_mask)} masks in {sub}")
        dataset_choices.append((img, mask))
        candidates_printed = True
if not dataset_choices:
    print("[ERROR] No data/*/images and data/*/masks found. Please organize and re-run.")
    # Print all possible existing folders and example files for manual debug:
    for sub in ["train", "val", "giramille_processed", "processed"]:
        base = DATA_ROOT / sub
        img = base / "images"
        mask = base / "masks"
        print(f"  [EXISTS] images: {img.exists()}, masks: {mask.exists()} at {img} / {mask}")
        if img.exists():
            print(f"    [EX] images: {[x.name for x in img.iterdir()]}")
        if mask.exists():
            print(f"    [EX] masks: {[x.name for x in mask.iterdir()]}")
    sys.exit(1)
if not candidates_printed:
    print("[INFO] No files printed, but dataset_choices found. (Unexpected)")
IMG_DIR, MASK_DIR = dataset_choices[0]

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_size=(256,256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.imgs = sorted(os.listdir(image_dir))
        self.transform = transform
        self.mask_size = mask_size
        self.valid_indices = self._find_valid_indices()
    def _find_valid_indices(self):
        valid = []
        for idx, fname in enumerate(self.imgs):
            img_path = self.image_dir / fname
            mask_path = self.mask_dir / fname
            try:
                img = Image.open(img_path)
                mask = Image.open(mask_path)
                # Try decompressing a small tile to trigger DecompressionBombError if any
                _ = img.getpixel((0,0))
                _ = mask.getpixel((0,0))
            except Exception as e:
                print(f"[SKIP] {fname}: unreadable or too large for memory. {e}")
                continue
            valid.append(idx)
        print(f"[INFO] Using {len(valid)} valid image/mask pairs (from {len(self.imgs)})")
        return valid
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_path = self.image_dir / self.imgs[real_idx]
        mask_path = self.mask_dir / self.imgs[real_idx]
        img = Image.open(img_path).convert('RGB').resize(self.mask_size, Image.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize(self.mask_size, Image.NEAREST)
        if self.transform:
            img = self.transform(img)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    def __len__(self):
        return len(self.valid_indices)

# --- Config ---
num_classes = 16
batch_size = 8
lr = 1e-3
epochs = 25
img_size = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
train_dataset = SegDataset(IMG_DIR, MASK_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = SmallUNet(in_ch=3, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader)
    for img, mask in progress:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1}: mean loss = {total_loss/len(train_loader):.4f}")
    if (epoch+1) % 5 == 0:
        torch.save({"model_state_dict": model.state_dict()}, f"smallunet_epoch{epoch+1}.pth")
torch.save({"model_state_dict": model.state_dict()}, "smallunet_best.pth")
print("Training finished, weights saved to smallunet_best.pth")
