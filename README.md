# Video Enhancer on RunPod

Upscale grainy old-phone videos to 1080p on a rented GPU, from a web page in your browser.

Two engines, selectable per job:

| Engine | What it is | Speed (RTX 4090, 480p → 1080p) | Best for |
|---|---|---|---|
| **Quality (SeedVR2)** | Video-native diffusion upscaler. Temporally stable, restores faces without a separate face model. | roughly real-time ÷ 5–20 (minutes per minute of video) | Final versions |
| **Fast (Real-ESRGAN + face restore)** | Per-frame Real-ESRGAN, then CodeFormer/GFPGAN on each detected face. | a few minutes per minute of video | Quick previews, clips with little motion |

Output is H.264 MP4 fitted inside 1920x1080 (or 1080x1920 for portrait), original audio kept.

## 1. Create the pod

1. RunPod → Pods → Deploy. Template **RunPod PyTorch 2.8.0** (image `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`).
2. GPU: **RTX 4090 24 GB** is enough for Fast and SeedVR2-3B. Use an **A40 / L40S 48 GB** if you want SeedVR2-7B fp16 without block swapping.
3. Container disk 20 GB, **Volume 100 GB** mounted at `/workspace`.
4. **Expose HTTP ports: `7860`**. Expose TCP ports: `22` (for scp; optional).
5. Deploy, wait for "Running".

## 2. First boot

Open the pod's web terminal (or SSH) and run:

```bash
cd /workspace
git clone https://github.com/<you>/enhance.git       # or unzip a copy sent with runpodctl
bash enhance/setup.sh                                 # ~10-15 min: venvs, SeedVR2 clone, weights (~8 GB)
/workspace/start.sh
```

Then open `https://<POD_ID>-7860.proxy.runpod.net`.

`setup.sh` is idempotent (skips finished steps; `--force` redoes everything) and logs to `/workspace/setup.log`. Everything it installs lives on the persistent volume, so after a pod stop/start you only need `/workspace/start.sh` again.

### Optional: auto-start on boot

Edit the pod → **Container Start Command**:

```
bash -c "[ -x /workspace/start.sh ] && nohup /workspace/start.sh >/workspace/app.log 2>&1 & /start.sh"
```

This keeps the template's own SSH/Jupyter start script and launches the UI in the background.

### No spare HTTP port? Use a tunnel

If the pod only exposes Jupyter's 8888 and you want to keep Jupyter, run the server on 7860 as usual and open a Cloudflare quick tunnel in a second terminal:

```bash
bash /workspace/enhance/tunnel.sh
```

It prints a public `https://<random>.trycloudflare.com` URL that goes straight to the enhancer. The URL changes each time the tunnel starts, and it is reachable by anyone who has it, so set `ENHANCE_AUTH` (below) first.

### Optional: password

Set `ENHANCE_AUTH="user:pass"` in the pod's environment variables, or put the line `ENHANCE_AUTH=user:pass` in `/workspace/.env` (SSH shells do not see RunPod env vars; `start.sh` reads this file). It protects both the UI and the upload/download API. Without it, anyone with the proxy URL can use the pod.

## 3. Daily use

1. **Upload** with the *Large file upload* control at the top of the page. It splits the file into 32 MB chunks because RunPod's proxy rejects single requests above ~100 MB. If the upload breaks, re-select the same file and it resumes.
   - Alternative: `runpodctl send video.mp4` locally, then `runpodctl receive <code>` in `/workspace/input` on the pod; or `scp -P <port> video.mp4 root@<ip>:/workspace/input/`.
2. Pick the file in the **Input video** dropdown (↻ Refresh if needed).
3. Choose the engine, adjust options, **Start**. Progress, log and ETA update every two seconds; you can close the tab and come back.
4. When done, preview inline or click **Download**. Finished files are also in `/workspace/output/`.

### SeedVR2 options

- **Model**: `3b_fp16` default (fits 24 GB at 1080p with batch 5). `7b_fp16` is sharper on faces but needs 48 GB or block swapping. fp8 variants trade a little quality for VRAM.
- **Batch size**: frames processed together (4n+1). Higher = better temporal consistency, more VRAM. 1 = frame-by-frame.
- **Color correction**: `lab` keeps colours faithful to the source.
- **Advanced**: on `CUDA out of memory`, lower batch size, enable *Offload* and set *Blocks to swap* to 16–32, or pick an fp8 model. *Chunk size* bounds system RAM use on long videos.

### Fast options

- **Upscaler**: `RealESRGAN_x4plus` (general), `x2plus` (softer), `realesr-general-x4v3` (small, fast).
- **Face model**: CodeFormer (default) or GFPGAN v1.4. **Fidelity** 0 = maximum restoration (can invent detail), 1 = closest to the input. 0.5–0.7 is a good range for real people.
- Faces are restored independently per frame, so slight flicker is possible; use SeedVR2 for the final version.

## 4. Layout on the pod

```
/workspace/enhance/        this repo
/workspace/venvs/fast      Python 3.11 venv: Gradio server + Real-ESRGAN/CodeFormer stack
/workspace/venvs/seedvr2   Python 3.12 venv for the SeedVR2 CLI
/workspace/SeedVR2/        numz/ComfyUI-SeedVR2_VideoUpscaler (pinned commit, used standalone)
/workspace/models/         weights (fast/, facexlib/, SEEDVR2/)
/workspace/input/          uploaded / copied source videos
/workspace/jobs/<id>/      job.json + log.txt per job (work files deleted on completion)
/workspace/output/         results: <name>_<engine>_1080p_<jobid>.mp4
```

Old uploads (>24 h) and job records (>7 days) are swept at server start.

## 5. Local development (Windows)

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe pytest numpy
.venv/Scripts/python.exe -m pytest
```

The tests cover resolution maths, SeedVR2 progress parsing and the chunked-upload protocol; they need no GPU or ffmpeg.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Upload stalls / 413 | Use the chunked uploader (top control), not the small-file box. |
| Page shows 524 | A handler took >100 s. Should not happen; report the log line. |
| `CUDA out of memory` in log | See SeedVR2 advanced options above, or Fast: set tile 512/256. |
| `missing weights ... run setup.sh` | `bash /workspace/enhance/setup.sh --force` |
| ffmpeg not found after pod restart | `start.sh` reinstalls it; if you started the server another way run `apt-get install -y ffmpeg`. |
| SeedVR2 output silent / wrong size | Expected from the CLI; the app re-muxes audio and rescales. Check `/workspace/jobs/<id>/log.txt`. |
| Want to update SeedVR2 | Bump `SEEDVR2_SHA` in `setup.sh`, then `setup.sh --force`. Check `inference_cli.py --help` for flag changes. |
