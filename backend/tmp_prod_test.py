from giramille_production import initialize_production_system
import os

# force CPU to avoid CUDA/device string issues
device = "cpu"
g = initialize_production_system(device)

res = g.generate_image(
    "white horse on a green grass with wooden compound",
    quality="balanced",
)

print(res)
if res.get("success"):
    os.makedirs("outputs/production", exist_ok=True)
    with open("outputs/production/tmp_output.png", "wb") as f:
        f.write(res["image"])
    print("Saved outputs/production/tmp_output2.png")