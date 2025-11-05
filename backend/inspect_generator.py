import inspect, traceback
try:
    # try both import locations depending on how you run scripts
    from giramille_production import initialize_production_system
except Exception:
    from backend.giramille_production import initialize_production_system

def main():
    try:
        g = initialize_production_system()
        print("generate_image signature:", inspect.signature(g.generate_image))
        keys = [a for a in dir(g) if any(x in a.lower() for x in ("guidance","step","scale","seed","negative","style","water","sampler","num","samples"))]
        print("filtered attributes:", keys)
        # optionally show small sample of reprs for config-like attrs
        for k in keys:
            try:
                val = getattr(g, k)
                print(f"{k} -> {type(val).__name__}")
            except Exception:
                print(f"{k} -> <unreadable>")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()