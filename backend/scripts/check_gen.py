import os
os.environ['GIRAMILLE_MODEL_DIR'] = r"D:\maitri\ai-image\giramille\backend\models\runwayml-stable-diffusion-v1-5\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
from backend.giramille_production import initialize_production_system
try:
    g = initialize_production_system(device='cpu')
    print('INIT_OK')
    print('device=', g.device)
    print('use_fp16=', g.use_fp16)
    print('has_generate_image=', callable(getattr(g, 'generate_image', None)))
    pipe = getattr(g, 'pipe', None)
    print('pipe_present=', pipe is not None)
    if pipe is not None:
        print('has_unet=', hasattr(pipe, 'unet'))
        print('has_text_encoder=', hasattr(pipe, 'text_encoder'))
        print('has_vae=', hasattr(pipe, 'vae'))
except Exception as e:
    import traceback
    traceback.print_exc()
    raise
