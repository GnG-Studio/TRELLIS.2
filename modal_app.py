import os
import subprocess

import modal

APP_NAME = "trellis2"
REPO_URL = "https://github.com/GnG-Studio/TRELLIS.2.git"
REPO_REF = "main"
REPO_DIR = "/opt/TRELLIS.2"
CACHE_DIR = "/cache"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("trellis2-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

# Match the CUDA 12.4 / PyTorch 2.6 stack used by the upstream TRELLIS.2 setup
# and by the earlier official Hugging Face Space build. Prebuilt wheels avoid
# compiling the CUDA extensions during a Modal image build.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libegl1",
        "libgl1",
        "libgles2",
        "libglib2.0-0",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        (
            "python -m pip install "
            "torch==2.6.0 torchvision==0.21.0 "
            "--index-url https://download.pytorch.org/whl/cu124"
        ),
        (
            "python -m pip install "
            "triton==3.2.0 "
            "pillow==12.0.0 "
            "imageio==2.37.2 "
            "imageio-ffmpeg==0.6.0 "
            "tqdm==4.67.1 "
            "easydict==1.13 "
            "opencv-python-headless==4.12.0.88 "
            "ninja "
            "trimesh==4.10.1 "
            "transformers==4.57.3 "
            "gradio==6.0.1 "
            "tensorboard pandas lpips zstandard "
            "kornia==0.8.2 timm==1.0.22 "
            "plyfile psutil safetensors huggingface_hub"
        ),
        (
            "python -m pip install "
            "'git+https://github.com/EasternJournalist/utils3d.git@"
            "9a4eb15e4021b67b12c460c7057d642626897ec8'"
        ),
        (
            "python -m pip install "
            "'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/"
            "flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' "
            "'https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
            "cumesh-0.0.1-cp310-cp310-linux_x86_64.whl' "
            "'https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
            "flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl' "
            "'https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
            "o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl' "
            "'https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
            "nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl' "
            "'https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/"
            "nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl'"
        ),
        f"git clone --recursive --depth=1 -b {REPO_REF} {REPO_URL} {REPO_DIR}",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HUGGINGFACE_HUB_CACHE": f"{CACHE_DIR}/huggingface/hub",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ATTN_BACKEND": "flash_attn",
            "GRADIO_SERVER_NAME": "0.0.0.0",
            "GRADIO_SERVER_PORT": "7860",
        }
    )
    .workdir(REPO_DIR)
)


@app.function(
    image=image,
    volumes={CACHE_DIR: hf_cache},
    secrets=[hf_secret],
    timeout=60 * 60,
    memory=32768,
)
def warm_cache():
    """Download the model files once into a persistent Modal Volume."""
    from huggingface_hub import snapshot_download

    repos = [
        "microsoft/TRELLIS.2-4B",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "briaai/RMBG-2.0",
    ]

    for repo_id in repos:
        print(f"Downloading {repo_id} ...")
        snapshot_download(repo_id)

    hf_cache.commit()
    print("Model cache committed to Modal Volume: trellis2-hf-cache")


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    volumes={CACHE_DIR: hf_cache},
    secrets=[hf_secret],
    timeout=60 * 60,
    startup_timeout=30 * 60,
    min_containers=0,
    max_containers=1,
    scaledown_window=120,
)
@modal.web_server(7860, startup_timeout=30 * 60)
def web():
    """Expose the original TRELLIS.2 Gradio UI on a Modal GPU."""
    env = os.environ.copy()
    env["GRADIO_SERVER_NAME"] = "0.0.0.0"
    env["GRADIO_SERVER_PORT"] = "7860"

    subprocess.Popen(
        ["python", "app.py"],
        cwd=REPO_DIR,
        env=env,
    )
