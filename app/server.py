"""FastAPI + Gradio server. Enhancement work never happens inside a Gradio handler: the RunPod proxy
kills any request over 100 s, so handlers only enqueue jobs and a gr.Timer polls their state."""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.engines import ENGINE_LABELS
from app.jobs import DONE, RUNNING, TERMINAL, JobManager
from app.upload import UploadError, UploadManager, list_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("enhance.server")

config.ensure_dirs()
jobs = JobManager()
uploads = UploadManager()
log.info("swept %d stale uploads, %d old jobs", uploads.sweep(), jobs.sweep())

STATIC_DIR = Path(__file__).parent / "static"
LABEL_TO_ENGINE = {v: k for k, v in ENGINE_LABELS.items()}

# --------------------------------------------------------------------------------------
# FastAPI routes (chunked upload, download)
# --------------------------------------------------------------------------------------
app = FastAPI(title="enhance")
demo: Optional[gr.Blocks] = None  # set below; needed by the auth dependency


gradio_app = None  # the mounted gradio App (holds the login token table); set in main()


def _find_gradio_app(root: FastAPI):
    """mount_gradio_app does not hand back the inner app, so locate it among the mounted routes."""
    from gradio.routes import App as GradioApp
    for route in root.routes:
        sub = getattr(route, "app", None)
        if isinstance(sub, GradioApp):
            return sub
    return None


def require_auth(request: Request) -> None:
    """When ENHANCE_AUTH is set, accept only requests carrying a valid Gradio login cookie.
    Gradio 6 names the cookie access-token-<cookie_id> (or access-token-unsecure-<cookie_id>)."""
    if not config.AUTH:
        return
    tokens = getattr(gradio_app, "tokens", None) or {}
    for name, value in request.cookies.items():
        if name.startswith("access-token") and value in tokens:
            return
    raise HTTPException(401, "login required")


@app.post("/api/upload/init", dependencies=[Depends(require_auth)])
async def upload_init(request: Request):
    body = await request.json()
    try:
        return uploads.init(str(body.get("filename", "")), int(body.get("size", 0)))
    except (UploadError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.put("/api/upload/chunk/{upload_id}/{index}", dependencies=[Depends(require_auth)])
async def upload_chunk(upload_id: str, index: int, request: Request):
    pieces = [piece async for piece in request.stream()]
    try:
        return uploads.write_chunk(upload_id, index, pieces)
    except UploadError as e:
        raise HTTPException(400, str(e))


@app.get("/api/upload/status/{upload_id}", dependencies=[Depends(require_auth)])
def upload_status(upload_id: str):
    try:
        return uploads.status(upload_id)
    except UploadError as e:
        raise HTTPException(404, str(e))


@app.post("/api/upload/complete/{upload_id}", dependencies=[Depends(require_auth)])
def upload_complete(upload_id: str):
    try:
        path = uploads.complete(upload_id)
    except UploadError as e:
        raise HTTPException(400, str(e))
    return {"path": str(path), "name": path.name}


@app.get("/api/download/{job_id}", dependencies=[Depends(require_auth)])
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job.state != DONE or not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(404, "no output for this job")
    p = Path(job.output_path)
    return FileResponse(str(p), media_type="video/mp4", filename=p.name)


@app.get("/api/health")
def health():
    return JSONResponse({"ok": True, "current_job": jobs.current, "jobs": len(jobs.jobs)})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --------------------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------------------
CSS = """
.hidden-box { display: none !important; }
.cu-wrap { padding: 8px 0; }
.cu-label { display: block; font-weight: 600; margin-bottom: 6px; }
.cu-input { display: block; margin-top: 6px; }
.cu-track { height: 10px; background: var(--border-color-primary, #ddd); border-radius: 5px; overflow: hidden; margin: 8px 0 4px; }
.cu-bar { height: 100%; width: 0; background: var(--color-accent, #f97316); transition: width .2s; }
.cu-status { font-size: 0.9em; opacity: .85; }
.prog-track { height: 14px; background: var(--border-color-primary, #ddd); border-radius: 7px; overflow: hidden; }
.prog-bar { height: 100%; background: var(--color-accent, #f97316); transition: width .5s; }
"""
HEAD = '<script src="/static/upload.js" defer></script>'


def _fmt_secs(s: Optional[float]) -> str:
    if s is None:
        return "–"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _disk_line() -> str:
    try:
        u = shutil.disk_usage(config.WORKSPACE)
        return f"disk free {u.free / 1e9:.1f} GB of {u.total / 1e9:.0f} GB"
    except OSError:
        return ""


def _input_choices() -> list:
    return [str(p) for p in list_inputs()]


def _history_rows() -> list:
    rows = []
    for j in jobs.list()[:50]:
        rows.append([
            time.strftime("%m-%d %H:%M", time.localtime(j.created_at)),
            Path(j.input_path).name, j.engine, j.state, f"{int(j.progress * 100)}%",
            _fmt_secs(j.elapsed) if j.started_at else "–",
            Path(j.output_path).name if j.output_path else (j.error or ""),
            j.id,
        ])
    return rows


def _progress_html(job) -> str:
    pct = int(job.progress * 100)
    return (f'<div class="prog-track"><div class="prog-bar" style="width:{pct}%"></div></div>'
            f'<div style="margin-top:4px">{pct}% – {job.stage}</div>')


def _status_md(job) -> str:
    lines = [f"**{Path(job.input_path).name}** · {ENGINE_LABELS.get(job.engine, job.engine)} · job `{job.id}`",
             f"State: **{job.state}** · elapsed {_fmt_secs(job.elapsed) if job.started_at else '–'}"
             + (f" · ETA {_fmt_secs(job.eta_s)}" if job.eta_s else "")]
    if job.error:
        lines.append(f"⚠️ {job.error}")
    lines.append(_disk_line())
    return "  \n".join(lines)


def _download_html(job) -> str:
    if job.state == DONE and job.output_path:
        name = Path(job.output_path).name
        return (f'<a href="/api/download/{job.id}" download="{name}" '
                f'style="font-weight:600">⬇ Download {name}</a> '
                f'<span style="opacity:.7">({Path(job.output_path).stat().st_size / 1e6:.0f} MB)</span>')
    return ""


def on_refresh_inputs(selected: Optional[str]):
    choices = _input_choices()
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def on_uploaded_path(path: str):
    """Fired by upload.js after a chunked upload completes."""
    choices = _input_choices()
    return gr.update(choices=choices, value=path if path in choices else (choices[0] if choices else None))


def on_small_file(file_path: Optional[str], selected: Optional[str]):
    """Gradio's own uploader (<90 MB): move the temp file into the input folder."""
    if not file_path:
        return on_refresh_inputs(selected)
    from app.upload import sanitize_filename, unique_path
    try:
        dst = unique_path(config.INPUT_DIR, sanitize_filename(Path(file_path).name))
    except UploadError as e:
        raise gr.Error(str(e))
    shutil.move(file_path, dst)
    return gr.update(choices=_input_choices(), value=str(dst))


def on_engine_change(label: str):
    is_seed = LABEL_TO_ENGINE.get(label) == "seedvr2"
    return gr.update(visible=is_seed), gr.update(visible=not is_seed)


def on_start(input_path: Optional[str], engine_label: str, crf: float,
             dit_model: str, batch_size: int, seed: float, color: str, temporal_overlap: float,
             chunk_size: float, blocks_to_swap: float, offload: bool, vae_tiled: bool,
             upscaler: str, face_restore: bool, face_model: str, fidelity: float, tile: int):
    if not input_path or not Path(input_path).exists():
        raise gr.Error("Pick an input video first (upload one or refresh the list).")
    if jobs.current is not None:
        raise gr.Error("A job is already running; wait for it to finish or cancel it.")
    engine = LABEL_TO_ENGINE.get(engine_label, "seedvr2")
    if engine == "seedvr2":
        opts = {"dit_model": dit_model, "batch_size": int(batch_size), "seed": int(seed),
                "color_correction": color, "temporal_overlap": int(temporal_overlap),
                "chunk_size": int(chunk_size), "blocks_to_swap": int(blocks_to_swap),
                "offload": bool(offload), "vae_decode_tiled": bool(vae_tiled), "crf": int(crf)}
    else:
        opts = {"upscaler": upscaler, "face_restore": bool(face_restore), "face_model": face_model,
                "fidelity": float(fidelity), "tile": int(tile), "crf": int(crf)}
    job = jobs.submit(Path(input_path), engine, opts)
    return job.id, {"job_id": job.id, "shown_state": None}


def on_cancel(job_id: Optional[str]):
    if job_id and jobs.cancel(job_id):
        return "cancel requested"
    return "nothing to cancel"


def on_delete(job_id: Optional[str], delete_output: bool):
    job = jobs.get(job_id)
    if job and job.state in TERMINAL:
        jobs.delete(job.id, delete_output=delete_output)
        return None, {"job_id": None, "shown_state": None}, gr.update(value=_history_rows())
    return job_id, gr.update(), gr.update(value=_history_rows())


def on_tick(job_id: Optional[str], view: dict):
    """Called every 2 s. Returns updates; the video is only (re)sent when the job's state changes."""
    view = dict(view or {})
    job = jobs.get(job_id)
    history = gr.update(value=_history_rows())
    if job is None:
        idle = f"Idle. {_disk_line()}" + (f" · running job {jobs.current}" if jobs.current else "")
        return idle, "", "", gr.update(), "", history, view
    status = _status_md(job)
    prog = _progress_html(job)
    logtxt = "\n".join(list(job.log)[-40:])
    changed = view.get("shown_state") != job.state or view.get("job_id") != job.id
    video = gr.update()
    if changed:
        view.update(job_id=job.id, shown_state=job.state)
        video = gr.update(value=job.output_path if (job.state == DONE and job.output_path) else None)
    return status, prog, logtxt, video, _download_html(job), history, view


def on_history_select(evt: gr.SelectData, rows):
    try:
        row = rows.iloc[evt.index[0]] if hasattr(rows, "iloc") else rows[evt.index[0]]
        job_id = str(row[-1] if not hasattr(row, "iloc") else row.iloc[-1])
    except (IndexError, TypeError, AttributeError):
        return gr.update(), gr.update()
    if jobs.get(job_id) is None:
        return gr.update(), gr.update()
    return job_id, {"job_id": job_id, "shown_state": None}


with gr.Blocks(title="Video Enhancer") as demo:
    gr.Markdown("# 🎞️ Video Enhancer\nUpscale old phone videos to 1080p with SeedVR2 (quality) or Real-ESRGAN + face restoration (fast).")
    job_state = gr.State(None)
    view_state = gr.State({"job_id": None, "shown_state": None})

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input")
            gr.HTML('<div id="chunk-uploader"></div>', padding=False)
            small_file = gr.File(label="Small file upload (< 90 MB)", file_types=list(config.ALLOWED_EXT), type="filepath")
            with gr.Row():
                input_dd = gr.Dropdown(label="Input video (/workspace/input)", choices=_input_choices(),
                                       value=(_input_choices() or [None])[0], scale=4)
                refresh_btn = gr.Button("↻ Refresh", scale=1)
            uploaded_path = gr.Textbox(elem_id="uploaded-path", elem_classes=["hidden-box"], label="uploaded-path")

            gr.Markdown("### 2. Engine")
            engine_radio = gr.Radio(list(ENGINE_LABELS.values()), value=ENGINE_LABELS["seedvr2"], label="Engine")
            with gr.Group(visible=True) as seed_group:
                dit_model = gr.Dropdown(config.SEEDVR2_DIT_MODELS, value=config.SEEDVR2_DEFAULT_DIT, label="SeedVR2 model")
                batch_size = gr.Dropdown(config.SEEDVR2_BATCH_SIZES, value=5, label="Batch size (frames; higher = more temporal consistency, more VRAM)")
                color = gr.Dropdown(config.SEEDVR2_COLOR_MODES, value="lab", label="Color correction")
                seed = gr.Number(value=42, label="Seed (-1 = random)", precision=0)
                with gr.Accordion("Advanced (memory)", open=False):
                    temporal_overlap = gr.Slider(0, 8, value=3, step=1, label="Temporal overlap between batches")
                    chunk_size = gr.Number(value=0, label="Chunk size (frames per pass; 0 = whole video). Set e.g. 200 for long clips or RAM errors", precision=0)
                    blocks_to_swap = gr.Slider(0, 36, value=0, step=1, label="Blocks to swap to CPU (0 = off; use 16-32 on OOM)")
                    offload = gr.Checkbox(value=False, label="Offload DiT/VAE to CPU between steps")
                    vae_tiled = gr.Checkbox(value=True, label="Tiled VAE decode (saves VRAM)")
            with gr.Group(visible=False) as fast_group:
                upscaler = gr.Dropdown(list(config.FAST_UPSCALERS), value="RealESRGAN_x4plus", label="Upscaler")
                face_restore = gr.Checkbox(value=True, label="Restore faces")
                face_model = gr.Dropdown(list(config.FACE_MODELS), value="CodeFormer", label="Face model")
                fidelity = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="CodeFormer fidelity (0 = sharper/more invented, 1 = closer to input)")
                tile = gr.Dropdown([0, 256, 512, 1024], value=0, label="Tile size (0 = auto)")
            crf = gr.Slider(12, 28, value=18, step=1, label="Output quality (CRF, lower = bigger/better)")
            with gr.Row():
                start_btn = gr.Button("▶ Start", variant="primary")
                cancel_btn = gr.Button("■ Cancel")

        with gr.Column(scale=1):
            gr.Markdown("### 3. Progress")
            status_md = gr.Markdown(f"Idle. {_disk_line()}")
            progress_html = gr.HTML("")
            log_box = gr.Textbox(label="Log", lines=12, max_lines=12, interactive=False)
            result_video = gr.Video(label="Result", interactive=False)
            download_html = gr.HTML("")
            with gr.Row():
                delete_output_cb = gr.Checkbox(value=False, label="also delete output file")
                delete_btn = gr.Button("🗑 Remove job from history")

    gr.Markdown("### History (click a row to view it)")
    history_df = gr.Dataframe(
        headers=["when", "file", "engine", "state", "progress", "elapsed", "output / error", "job"],
        datatype=["str"] * 8, value=_history_rows(), interactive=False, wrap=True)

    timer = gr.Timer(2.0)

    # wiring
    refresh_btn.click(on_refresh_inputs, inputs=[input_dd], outputs=[input_dd])
    uploaded_path.change(on_uploaded_path, inputs=[uploaded_path], outputs=[input_dd])
    small_file.upload(on_small_file, inputs=[small_file, input_dd], outputs=[input_dd])
    engine_radio.change(on_engine_change, inputs=[engine_radio], outputs=[seed_group, fast_group])
    start_btn.click(
        on_start,
        inputs=[input_dd, engine_radio, crf, dit_model, batch_size, seed, color, temporal_overlap,
                chunk_size, blocks_to_swap, offload, vae_tiled, upscaler, face_restore, face_model, fidelity, tile],
        outputs=[job_state, view_state])
    cancel_btn.click(on_cancel, inputs=[job_state], outputs=[status_md])
    delete_btn.click(on_delete, inputs=[job_state, delete_output_cb], outputs=[job_state, view_state, history_df])
    history_df.select(on_history_select, inputs=[history_df], outputs=[job_state, view_state])
    timer.tick(on_tick, inputs=[job_state, view_state],
               outputs=[status_md, progress_html, log_box, result_video, download_html, history_df, view_state])


def build_app() -> FastAPI:
    """Mount the Gradio UI onto the FastAPI app and remember the inner Gradio app for auth checks."""
    global app, gradio_app
    auth = None
    if config.AUTH and ":" in config.AUTH:
        user, pw = config.AUTH.split(":", 1)
        auth = (user, pw)
    elif config.AUTH:
        log.warning("ENHANCE_AUTH must look like user:pass; ignoring it")
    app = gr.mount_gradio_app(
        app, demo, path="/", auth=auth,
        allowed_paths=[str(config.OUTPUT_DIR)],
        max_file_size=config.SMALL_UPLOAD_LIMIT,
        css=CSS, head=HEAD, theme=gr.themes.Soft(),
    )
    gradio_app = _find_gradio_app(app)
    if auth and gradio_app is None:
        raise RuntimeError("could not locate the mounted Gradio app; auth would lock out the API")
    log.info("auth %s", "enabled" if auth else "disabled")
    return app


def main() -> None:
    build_app()
    log.info("serving on 0.0.0.0:%d (workspace=%s)", config.SERVER_PORT, config.WORKSPACE)
    uvicorn.run(app, host="0.0.0.0", port=config.SERVER_PORT, timeout_keep_alive=120, log_level="info")


if __name__ == "__main__":
    main()
