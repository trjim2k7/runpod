"""Paths, limits and model tables. Everything persistent lives under WORKSPACE."""
from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("ENHANCE_WORKSPACE", "/workspace"))

INPUT_DIR = WORKSPACE / "input"
UPLOAD_DIR = WORKSPACE / "uploads"
JOBS_DIR = WORKSPACE / "jobs"
OUTPUT_DIR = WORKSPACE / "output"
MODELS_DIR = WORKSPACE / "models"
FAST_MODELS_DIR = MODELS_DIR / "fast"
FACEXLIB_MODELS_DIR = MODELS_DIR / "facexlib"
SEEDVR2_MODELS_DIR = MODELS_DIR / "SEEDVR2"
HF_HOME = WORKSPACE / "hf"

SEEDVR2_REPO = WORKSPACE / "SeedVR2"
SEEDVR2_PYTHON = WORKSPACE / "venvs" / "seedvr2" / "bin" / "python"

# Output box: fit inside 1920x1080 (landscape) or 1080x1920 (portrait).
TARGET_LONG = 1920
TARGET_SHORT = 1080

CHUNK_SIZE = 32 * 1024 * 1024          # per-request upload chunk; RunPod proxy caps bodies near 100 MB
SMALL_UPLOAD_LIMIT = "90mb"            # Gradio's own uploader, for convenience only
ALLOWED_EXT = {".mp4", ".mov", ".3gp", ".avi", ".mkv", ".m4v", ".webm", ".mts", ".wmv"}

UPLOAD_MAX_AGE_H = 24
JOB_MAX_AGE_DAYS = 7

SERVER_PORT = int(os.environ.get("ENHANCE_PORT", "7860"))
AUTH = os.environ.get("ENHANCE_AUTH")  # "user:pass" to protect the UI and API

SEEDVR2_DIT_MODELS = [
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
]
SEEDVR2_DEFAULT_DIT = SEEDVR2_DIT_MODELS[0]
SEEDVR2_VAE = "ema_vae_fp16.safetensors"
SEEDVR2_COLOR_MODES = ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"]
SEEDVR2_BATCH_SIZES = [1, 5, 9, 13, 17, 21]

# name -> (filename, download url)
FAST_UPSCALERS = {
    "RealESRGAN_x4plus": (
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "RealESRGAN_x2plus": (
        "RealESRGAN_x2plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
    "realesr-general-x4v3": (
        "realesr-general-x4v3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ),
}
FACE_MODELS = {
    "CodeFormer": (
        "codeformer.pth",
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    ),
    "GFPGANv1.4": (
        "GFPGANv1.4.pth",
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    ),
}
FACEXLIB_WEIGHTS = {
    "detection_Resnet50_Final.pth":
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing_parsenet.pth":
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}


def ensure_dirs() -> None:
    for d in (INPUT_DIR, UPLOAD_DIR, JOBS_DIR, OUTPUT_DIR, FAST_MODELS_DIR,
              FACEXLIB_MODELS_DIR, SEEDVR2_MODELS_DIR, HF_HOME):
        d.mkdir(parents=True, exist_ok=True)
