"""
Simple helper to download a model repo locally using huggingface_hub.snapshot_download.
Usage:
  python fetch_model_weights.py --model runwayml/stable-diffusion-v1-5 --dest ../models/runwayml-stable-diffusion-v1-5
If you want to download private models, set HF_TOKEN env var or pass --token.
"""
import argparse
import os
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("huggingface_hub is not installed or importing it failed. Install with: pip install huggingface-hub")
    print("Import error:", e)
    import sys
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hugging Face repo id or local path")
    p.add_argument("--dest", required=True, help="Local destination folder to place model files")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face access token (or set HF_TOKEN env)")
    args = p.parse_args()

    model = args.model
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading model {model} into {dest} (this may be large)")
    try:
        repo_local_dir = snapshot_download(repo_id=model, cache_dir=dest, repo_type="model", use_auth_token=args.token)
        print("Snapshot downloaded to:", repo_local_dir)
    except Exception as e:
        print("Download failed:", e)
        sys.exit(2)

if __name__ == '__main__':
    main()
