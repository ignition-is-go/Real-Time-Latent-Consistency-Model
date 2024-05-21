---
title: RTLCM-Gfx2Cuda
---

# Real-Time Latent Consistency Model

## Running Locally

You need Windows, DX11, CUDA, and Python 3.10

## Installation and Usage Example (Using Conda)

```bash
conda create -n rtlcm python=3.10
conda activate rtlcm
python3 -m pip install --upgrade pip
install.bat
python3 main.py --reload --pipeline img2imgSDXL-Lightning
```

## Gfx2Cuda

This implementation of RTLCM uses Gfx2Cuda to write tensors directly to texture buffers. Currently, Gfx2Cuda only supports DX11.
https://github.com/SvenDH/gfx2cuda

# Available Pipelines

See ./pipelines
When using the --pipeline flag, use the name of the file without the .py extension
E.g. --pipeline img2imgSDXL-Lightning

### Setting environment variables

- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port number (default: 7860)
- `--reload`: Reload code on change
- `--max-queue-size`: Maximum queue size (optional)
- `--timeout`: Timeout period (optional)
- `--safety-checker`: Enable Safety Checker (optional)
- `--torch-compile`: Use Torch Compile
- `--use-taesd` / `--no-taesd`: Use Tiny Autoencoder
- `--pipeline`: Pipeline to use (default: "txt2img")
- `--ssl-certfile`: SSL Certificate File (optional)
- `--ssl-keyfile`: SSL Key File (optional)
- `--debug`: Print Inference time
- `--compel`: Compel option
- `--sfast`: Enable Stable Fast
- `--onediff`: Enable OneDiff

If you run using `bash build-run.sh` you can set `PIPELINE` variables to choose the pipeline you want to run

```bash
PIPELINE=txt2imgLoraSDXL bash build-run.sh
```

and setting environment variables

```bash
TIMEOUT=120 SAFETY_CHECKER=True MAX_QUEUE_SIZE=4 python server/main.py --reload --pipeline txt2imgLoraSDXL
```

If you're running locally and want to test it on Mobile Safari, the webserver needs to be served over HTTPS, or follow this instruction on my [comment](https://github.com/radames/Real-Time-Latent-Consistency-Model/issues/17#issuecomment-1811957196)

```bash
openssl req -newkey rsa:4096 -nodes -keyout key.pem -x509 -days 365 -out certificate.pem
python server/main.py --reload --ssl-certfile=certificate.pem --ssl-keyfile=key.pem
```

## Docker

You need NVIDIA Container Toolkit for Docker, defaults to `controlnet``

```bash
docker build -t lcm-live .
docker run -ti -p 7860:7860 --gpus all lcm-live
```

reuse models data from host to avoid downloading them again, you can change `~/.cache/huggingface` to any other directory, but if you use hugingface-cli locally, you can share the same cache

```bash
docker run -ti -p 7860:7860 -e HF_HOME=/data -v ~/.cache/huggingface:/data  --gpus all lcm-live
```

or with environment variables

```bash
docker run -ti -e PIPELINE=txt2imgLoraSDXL -p 7860:7860 --gpus all lcm-live
```

# Demo on Hugging Face

- [radames/Real-Time-Latent-Consistency-Model](https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model)
- [radames/Real-Time-SD-Turbo](https://huggingface.co/spaces/radames/Real-Time-SD-Turbo)
- [latent-consistency/Real-Time-LCM-ControlNet-Lora-SD1.5](https://huggingface.co/spaces/latent-consistency/Real-Time-LCM-ControlNet-Lora-SD1.5)
- [latent-consistency/Real-Time-LCM-Text-to-Image-Lora-SD1.5](https://huggingface.co/spaces/latent-consistency/Real-Time-LCM-Text-to-Image-Lora-SD1.5)
- [radames/Real-Time-Latent-Consistency-Model-Text-To-Image](https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model-Text-To-Image)

https://github.com/radames/Real-Time-Latent-Consistency-Model/assets/102277/c4003ac5-e7ff-44c0-97d3-464bb659de70
