from giramille_production import initialize_production_system
import torch, os

device = "cuda" if torch.cuda.is_available() else "cpu"
g = initialize_production_system(device)
res = g.generate_image("A colorful sunset over the ocean in Giramille style, vivid saturation", quality="balanced")

print("result keys:", list(res.keys()))
if res.get("success"):
    os.makedirs("outputs/production", exist_ok=True)
    with open("outputs/production/test_output.png", "wb") as f:
        f.write(res["image"])
    print("Saved outputs/production/test_output.png")
else:
    print("Generation failed:", res.get("error"))