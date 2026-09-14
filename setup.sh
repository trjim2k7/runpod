#!/usr/bin/env bash
# One-time (idempotent) installer for a RunPod PyTorch pod. Everything lands on /workspace so it
# survives pod stop/start. Safe to re-run; pass --force to redo all steps.
set -euo pipefail

SETUP_VERSION="2026-09-13.1"
WORKSPACE="${ENHANCE_WORKSPACE:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDVR2_SHA="4490bd1f482e026674543386bb2a4d176da245b9"   # numz/ComfyUI-SeedVR2_VideoUpscaler, 2025-12-24 (v2.5.24)
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

export PATH="$WORKSPACE/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/uv-python"
export UV_CACHE_DIR="$WORKSPACE/.uv-cache"
export HF_HOME="$WORKSPACE/hf"
export DEBIAN_FRONTEND=noninteractive

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

mkdir -p "$WORKSPACE"
LOG="$WORKSPACE/setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== setup.sh $SETUP_VERSION started $(date -u +%FT%TZ) (repo: $REPO_DIR)"

if [[ $FORCE -eq 0 && -f "$WORKSPACE/.setup_done" && "$(cat "$WORKSPACE/.setup_done")" == "$SETUP_VERSION" ]]; then
  echo "setup already complete for version $SETUP_VERSION (use --force to redo). Run: $WORKSPACE/start.sh"
  exit 0
fi

step() { echo; echo "--- $*"; }

# 1. system packages (container disk; start.sh re-checks ffmpeg on every boot)
step "apt packages"
if ! command -v ffmpeg >/dev/null || ! command -v aria2c >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends ffmpeg git aria2 libgl1 libglib2.0-0 >/dev/null
fi
ffmpeg -version | head -1

# 2. uv + interpreters (persistent)
step "uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORKSPACE/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv --version
uv python install 3.11 3.12

# 3. fast/app venv (Python 3.11): torch cu128 first, then the stack
step "venv: fast/app (py3.11)"
FAST_VENV="$WORKSPACE/venvs/fast"
[[ -x "$FAST_VENV/bin/python" ]] || uv venv "$FAST_VENV" --python 3.11
FPY="$FAST_VENV/bin/python"
uv pip install --python "$FPY" torch torchvision --index-url "$TORCH_INDEX"
uv pip install --python "$FPY" -r "$REPO_DIR/requirements-app.txt"
uv pip install --python "$FPY" -r "$REPO_DIR/requirements-fast.txt"
# facexlib pulls opencv-python (GUI build); install it without deps and add what it needs.
uv pip install --python "$FPY" --no-deps facexlib==0.3.0
uv pip install --python "$FPY" filterpy numba scipy Pillow tqdm
"$FPY" - <<'PY'
import torch, spandrel, spandrel_extra_arches, facexlib, gradio, cv2
print("fast venv ok: torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "| gradio", gradio.__version__, "| spandrel", spandrel.__version__)
PY

# 4. SeedVR2 venv (Python 3.12) + pinned clone
step "venv: seedvr2 (py3.12)"
SEED_VENV="$WORKSPACE/venvs/seedvr2"
[[ -x "$SEED_VENV/bin/python" ]] || uv venv "$SEED_VENV" --python 3.12
SPY="$SEED_VENV/bin/python"
SEED_DIR="$WORKSPACE/SeedVR2"
if [[ ! -d "$SEED_DIR/.git" ]]; then
  git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$SEED_DIR"
fi
git -C "$SEED_DIR" fetch --quiet origin
git -C "$SEED_DIR" checkout --quiet "$SEEDVR2_SHA"
uv pip install --python "$SPY" torch torchvision --index-url "$TORCH_INDEX"
uv pip install --python "$SPY" -r "$SEED_DIR/requirements.txt" huggingface_hub
# requirements.txt may have re-resolved torch from PyPI; make sure the CUDA build is in place.
if ! "$SPY" -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)"; then
  echo "torch lost CUDA support; reinstalling cu128 build"
  uv pip install --python "$SPY" --reinstall torch torchvision --index-url "$TORCH_INDEX"
fi
"$SPY" - <<'PY'
import torch
print("seedvr2 venv ok: torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
"$SPY" "$SEED_DIR/inference_cli.py" --help >/dev/null && echo "inference_cli.py --help ok"

# 5. weights
step "weights"
MODELS="$WORKSPACE/models"
mkdir -p "$MODELS/fast" "$MODELS/facexlib" "$MODELS/SEEDVR2"
fetch() { # fetch <dir> <url>
  local dir="$1" url="$2" name; name="$(basename "$url")"
  if [[ -s "$dir/$name" ]]; then echo "have $name"; return; fi
  aria2c -q -x8 -s8 -c --allow-overwrite=true -d "$dir" -o "$name" "$url" || curl -L --retry 3 -o "$dir/$name" "$url"
  echo "downloaded $name"
}
fetch "$MODELS/fast" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fetch "$MODELS/fast" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
fetch "$MODELS/fast" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
fetch "$MODELS/fast" https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
fetch "$MODELS/fast" https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
fetch "$MODELS/facexlib" https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
fetch "$MODELS/facexlib" https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

# SeedVR2 3B fp16 + VAE (~6 GB) so the first job does not stall; other models auto-download on demand.
"$SPY" - <<PY
from huggingface_hub import hf_hub_download
for f in ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"):
    p = hf_hub_download("numz/SeedVR2_comfyUI", f, local_dir="$MODELS/SEEDVR2")
    print("have", p)
PY

# 6. runtime dirs + start script
step "start.sh"
mkdir -p "$WORKSPACE/input" "$WORKSPACE/uploads" "$WORKSPACE/jobs" "$WORKSPACE/output" "$HF_HOME"
sed -e "s|@WORKSPACE@|$WORKSPACE|g" -e "s|@REPO_DIR@|$REPO_DIR|g" "$REPO_DIR/start.sh.tpl" > "$WORKSPACE/start.sh"
chmod +x "$WORKSPACE/start.sh"
echo "$SETUP_VERSION" > "$WORKSPACE/.setup_done"
echo
echo "=== setup complete. Start the server with:  $WORKSPACE/start.sh"
echo "    then open https://<POD_ID>-7860.proxy.runpod.net"
