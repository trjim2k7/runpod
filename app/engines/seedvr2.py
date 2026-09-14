"""Quality engine: runs the SeedVR2 standalone CLI in its own venv, then re-muxes audio."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from app import config
from app.ffmpeg import mux_audio, normalize_cfr, probe, target_resolution, validate_output

# Progress lines printed by inference_cli.py / its ffmpeg writer. Kept tolerant on purpose.
_WRITTEN_RE = re.compile(r"Written\s+(\d+)\s*/\s*(\d+)\s+frames", re.I)
_BATCH_RE = re.compile(r"Upscaling batch\s+(\d+)\s*/\s*(\d+)", re.I)   # "[19:43:11] 🎬 Upscaling batch 80/165"
_CHUNK_RE = re.compile(r"Chunk\s+(\d+)\s*/\s*(\d+)", re.I)
_PCT_RE = re.compile(r"(\d{1,3})%\|")            # tqdm style "45%|####"
_FRAC_RE = re.compile(r"\b(\d+)/(\d+)\b")
_IGNORE = ("EulerSampler",)                      # per-batch one-step sampler bar: always 0% or 100%


def parse_progress(line: str) -> Optional[float]:
    """Return a 0..1 fraction if the line carries progress, else None."""
    if any(tag in line for tag in _IGNORE):
        return None
    m = _WRITTEN_RE.search(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        return done / total if total > 0 else None
    m = _BATCH_RE.search(line)
    if m:
        idx, total = int(m.group(1)), int(m.group(2))
        return (idx - 1) / total if total > 0 and idx >= 1 else None
    m = _CHUNK_RE.search(line)
    if m:
        idx, total = int(m.group(1)), int(m.group(2))
        return (idx - 1) / total if total > 0 and idx >= 1 else None
    m = _PCT_RE.search(line)
    if m:
        return int(m.group(1)) / 100.0
    if "it/s" in line or "s/it" in line:
        m = _FRAC_RE.search(line)
        if m and int(m.group(2)) > 0:
            return int(m.group(1)) / int(m.group(2))
    return None


class SeedVR2Engine:
    name = "seedvr2"

    def build_command(self, cfr_input: Path, out_file: Path, short_side: int, opts: dict) -> list:
        cmd = [str(config.SEEDVR2_PYTHON), str(config.SEEDVR2_REPO / "inference_cli.py"), str(cfr_input),
               "--output", str(out_file), "--output_format", "mp4",
               "--resolution", str(short_side),
               "--batch_size", str(int(opts.get("batch_size", 5))),
               "--dit_model", str(opts.get("dit_model", config.SEEDVR2_DEFAULT_DIT)),
               "--model_dir", str(config.SEEDVR2_MODELS_DIR),
               "--color_correction", str(opts.get("color_correction", "lab")),
               "--temporal_overlap", str(int(opts.get("temporal_overlap", 3))),
               "--video_backend", "ffmpeg",
               "--attention_mode", str(opts.get("attention_mode", "sdpa")),
               "--cuda_device", "0"]
        seed = int(opts.get("seed", 42))
        if seed < 0:
            seed = int(time.time()) % 2_000_000_000
        cmd += ["--seed", str(seed)]
        if opts.get("vae_decode_tiled", True):
            cmd += ["--vae_decode_tiled"]
        chunk = int(opts.get("chunk_size", 0))
        if chunk > 0:
            cmd += ["--chunk_size", str(chunk)]
        blocks = int(opts.get("blocks_to_swap", 0))
        if blocks > 0 or opts.get("offload"):
            cmd += ["--dit_offload_device", "cpu", "--vae_offload_device", "cpu"]
        if blocks > 0:
            cmd += ["--blocks_to_swap", str(blocks)]
        if opts.get("debug"):
            cmd += ["--debug"]
        return cmd

    def run(self, job, on_progress, cancel_event: threading.Event) -> Path:
        src = Path(job.input_path)
        info = probe(src)
        tw, th = target_resolution(info.width, info.height)
        short_side = min(tw, th)
        job.log_line(f"source {info.width}x{info.height} @ {float(info.fps):.3f} fps, {info.duration:.1f}s, "
                     f"audio={info.audio_codec}; target {tw}x{th}")

        work = job.work_dir
        cfr = work / "input_cfr.mp4"
        on_progress(0.0, "normalising input")
        normalize_cfr(src, cfr, info.fps_str)
        if cancel_event.is_set():
            raise RuntimeError("cancelled")

        out_noaudio = work / "seedvr2_out.mp4"
        cmd = self.build_command(cfr, out_noaudio, short_side, job.options)
        job.log_line("$ " + " ".join(cmd))
        on_progress(0.0, "loading SeedVR2 (first run downloads the model)")

        env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HOME=str(config.HF_HOME),
                   HF_HUB_ENABLE_HF_TRANSFER="0")
        proc = subprocess.Popen(cmd, cwd=str(config.SEEDVR2_REPO), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace",
                                start_new_session=True)
        job.proc = proc
        watcher = threading.Thread(target=self._kill_on_cancel, args=(proc, cancel_event), daemon=True)
        watcher.start()

        oom = False
        assert proc.stdout is not None
        # readline() rather than iterating the pipe: iteration read-aheads in 8 KB blocks and lags live output.
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            job.log_line(line)
            if "out of memory" in line.lower():
                oom = True
            frac = parse_progress(line)
            if frac is not None:
                on_progress(0.02 + 0.93 * frac, "upscaling")
        rc = proc.wait()
        job.proc = None
        if cancel_event.is_set():
            raise RuntimeError("cancelled")
        if rc != 0:
            hint = " (CUDA out of memory: lower batch_size to 1, pick an fp8 model, or add blocks_to_swap)" if oom else ""
            raise RuntimeError(f"SeedVR2 exited with code {rc}{hint}")

        produced = self._find_output(out_noaudio, work, exclude={cfr})
        if produced is None:
            raise RuntimeError("SeedVR2 finished but no output video was found in the work directory")
        job.log_line(f"SeedVR2 wrote {produced}")

        on_progress(0.96, "muxing audio")
        got = probe(produced)
        scale_to = None if (got.width, got.height) == (tw, th) else (tw, th)
        if scale_to:
            job.log_line(f"rescaling {got.width}x{got.height} -> {tw}x{th} during mux")
        final = work / "output.mp4"
        mux_audio(produced, src, final, info.audio_codec, scale_to=scale_to, crf=int(job.options.get("crf", 18)))
        validate_output(final, info.duration)
        on_progress(1.0, "done")
        return final

    @staticmethod
    def _kill_on_cancel(proc: subprocess.Popen, cancel_event: threading.Event) -> None:
        while proc.poll() is None:
            if cancel_event.wait(0.5):
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, AttributeError, PermissionError):
                    proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, AttributeError, PermissionError):
                        proc.kill()
                return

    @staticmethod
    def _find_output(expected: Path, work: Path, exclude: set) -> Optional[Path]:
        if expected.exists() and expected.stat().st_size > 0:
            return expected
        candidates = [p for p in work.rglob("*.mp4") if p not in exclude and p.stat().st_size > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
