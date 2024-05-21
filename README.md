---
title: RTLCM-Gfx2Cuda
---

# Real-Time Latent Consistency Model

A fork of https://github.com/radames/Real-Time-Latent-Consistency-Model

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

# Demo on Hugging Face

- [radames/Real-Time-Latent-Consistency-Model](https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model)
- [radames/Real-Time-SD-Turbo](https://huggingface.co/spaces/radames/Real-Time-SD-Turbo)
- [latent-consistency/Real-Time-LCM-ControlNet-Lora-SD1.5](https://huggingface.co/spaces/latent-consistency/Real-Time-LCM-ControlNet-Lora-SD1.5)
- [latent-consistency/Real-Time-LCM-Text-to-Image-Lora-SD1.5](https://huggingface.co/spaces/latent-consistency/Real-Time-LCM-Text-to-Image-Lora-SD1.5)
- [radames/Real-Time-Latent-Consistency-Model-Text-To-Image](https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model-Text-To-Image)

https://github.com/radames/Real-Time-Latent-Consistency-Model/assets/102277/c4003ac5-e7ff-44c0-97d3-464bb659de70
