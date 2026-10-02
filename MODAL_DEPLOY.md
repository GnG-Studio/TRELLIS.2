# Deploy TRELLIS.2 on Modal

This branch adds a Modal launcher without changing the TRELLIS.2 inference code.

## What it does

- Runs the existing Gradio `app.py`.
- Uses an NVIDIA L4 (24 GB VRAM) by default.
- Keeps at most one GPU container alive.
- Scales to zero after idle time.
- Stores Hugging Face model files in a persistent Modal Volume.
- Uses the CUDA 12.4 / PyTorch 2.6 dependency stack compatible with this TRELLIS.2 checkout.

## 1. Prerequisites

Install and authenticate Modal:

```powershell
py -m pip install -U modal
modal setup
```

You also need a Hugging Face read token. Before creating it, accept access terms for the gated models used by the TRELLIS.2 pipeline:

- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `briaai/RMBG-2.0`

Do not commit your Hugging Face token to this repository.

## 2. Create the Modal secret

```powershell
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

## 3. Warm the persistent model cache

Run once:

```powershell
modal run modal_app.py::warm_cache
```

This downloads TRELLIS.2-4B plus its DINOv3 and RMBG dependencies into the persistent Modal Volume `trellis2-hf-cache`.

## 4. Deploy

```powershell
modal deploy modal_app.py
```

Modal will print a public URL for the Gradio application.

## Recommended first test

Start with:

```text
Resolution: 512
Texture size: 1024 or 2048
```

The L4 has 24 GB VRAM, which is the minimum class of GPU targeted here. If higher resolutions repeatedly run out of memory, change this line in `modal_app.py`:

```python
gpu="L4"
```

to:

```python
gpu="L40S"
```

The L40S has substantially more VRAM but costs more.

## Cost controls

The web function is configured with:

```python
min_containers=0
max_containers=1
scaledown_window=120
```

So the GPU can scale to zero when idle and the deployment cannot automatically create multiple GPU containers.

## Useful commands

View logs:

```powershell
modal app logs trellis2
```

Stop the deployed app:

```powershell
modal app stop trellis2
```

Inspect the cache volume:

```powershell
modal volume ls trellis2-hf-cache
```

Delete the cache if you want to start over:

```powershell
modal volume delete trellis2-hf-cache
```
