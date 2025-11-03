from giramille_production import initialize_production_system
import inspect, traceback, base64, os

PROMPT = "A fairy tale castle on top of a hill during sunset"

try:
    g = initialize_production_system()  # adapt if you need a device arg
    sig = inspect.signature(g.generate_image)
    print("generate_image signature:", sig)

    # try several call forms
    attempts = []
    try:
        attempts.append(("kwargs", g.generate_image(prompt=PROMPT)))
    except Exception as e:
        attempts.append(("kwargs failed", str(e)))
    try:
        attempts.append(("positional_prompt", g.generate_image(PROMPT)))
    except Exception as e:
        attempts.append(("positional_prompt failed", str(e)))
    try:
        attempts.append(("prompt_w_h_q", g.generate_image(PROMPT, 512, 512, "balanced")))
    except Exception as e:
        attempts.append(("prompt_w_h_q failed", str(e)))

    for name, res in attempts:
        print("ATTEMPT:", name)
        if isinstance(res, dict):
            print("  keys:", list(res.keys()))
            if res.get("success") and res.get("image"):
                out = os.path.join(os.path.dirname(__file__), "outputs", "production")
                os.makedirs(out, exist_ok=True)
                out_path = os.path.join(out, "test_out.png")
                with open(out_path, "wb") as f:
                    f.write(res["image"])
                print("  saved image to:", out_path)
                break
            else:
                print("  failed:", res.get("error"))
        else:
            print("  exception:", res)

except Exception:
    traceback.print_exc()