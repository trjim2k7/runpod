"""ffmpeg/ffprobe helpers: probing, CFR normalisation, raw frame decode/encode, audio mux."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from app.config import TARGET_LONG, TARGET_SHORT

COPYABLE_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3"}


class FFmpegError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    width: int            # post-rotation
    height: int           # post-rotation
    fps: Fraction
    duration: float
    nb_frames: int
    audio_codec: Optional[str]
    rotation: int

    @property
    def fps_str(self) -> str:
        return f"{self.fps.numerator}/{self.fps.denominator}"

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


def run(cmd: list, timeout: Optional[float] = None) -> str:
    """Run a command, return stdout, raise FFmpegError with stderr on failure."""
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        tail = p.stderr.strip().splitlines()[-15:]
        raise FFmpegError(f"{cmd[0]} failed ({p.returncode}):\n" + "\n".join(tail))
    return p.stdout


def _parse_fraction(s: str) -> Optional[Fraction]:
    try:
        if "/" in s:
            n, d = s.split("/")
            n, d = int(n), int(d)
            if n <= 0 or d <= 0:
                return None
            return Fraction(n, d)
        f = Fraction(s)
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _stream_rotation(stream: dict) -> int:
    rot = 0
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            rot = int(float(tags["rotate"]))
        except ValueError:
            rot = 0
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rot = int(float(sd["rotation"]))
            except ValueError:
                pass
    return rot % 360


def probe(path: Path) -> VideoInfo:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path])
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"no video stream in {path}")
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    w, h = int(video["width"]), int(video["height"])
    rotation = _stream_rotation(video)
    if rotation in (90, 270):
        w, h = h, w

    fps = (_parse_fraction(video.get("avg_frame_rate", "0/0"))
           or _parse_fraction(video.get("r_frame_rate", "0/0"))
           or Fraction(30))
    if fps > 240:  # bogus avg rate on some phone files
        fps = _parse_fraction(video.get("r_frame_rate", "0/0")) or Fraction(30)

    duration = 0.0
    for src in (video.get("duration"), (data.get("format") or {}).get("duration")):
        if src:
            try:
                duration = float(src)
                break
            except ValueError:
                pass
    nb = 0
    if video.get("nb_frames"):
        try:
            nb = int(video["nb_frames"])
        except ValueError:
            nb = 0
    if nb <= 0 and duration > 0:
        nb = int(round(duration * float(fps)))

    return VideoInfo(w, h, fps, duration, nb, audio.get("codec_name") if audio else None, rotation)


def target_resolution(w: int, h: int, long_side: int = TARGET_LONG, short_side: int = TARGET_SHORT,
                      allow_downscale: bool = False) -> tuple[int, int]:
    """Fit (w, h) into long_side x short_side keeping aspect. Never shrinks unless allowed. Even dims."""
    if w <= 0 or h <= 0:
        raise ValueError("bad dimensions")
    long_in, short_in = max(w, h), min(w, h)
    scale = min(long_side / long_in, short_side / short_in)
    if scale < 1 and not allow_downscale:
        scale = 1.0
    tw, th = int(round(w * scale)), int(round(h * scale))
    return max(2, tw - tw % 2), max(2, th - th % 2)


def normalize_cfr(src: Path, dst: Path, fps_str: str, crf: int = 14) -> None:
    """Re-encode to constant-frame-rate H.264 (bakes rotation, drops audio). Used before SeedVR2."""
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-fps_mode", "cfr", "-r", fps_str,
         "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-an", dst])


class FrameDecoder:
    """Yields BGR uint8 frames at constant frame rate. Use as a context manager so ffmpeg is reaped."""

    def __init__(self, path: Path, width: int, height: int, fps_str: str):
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "cfr", "-r", fps_str,
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def __enter__(self) -> "FrameDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def frames(self) -> Iterator[np.ndarray]:
        assert self.proc.stdout is not None
        while True:
            buf = self.proc.stdout.read(self.frame_bytes)
            if len(buf) < self.frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


class FrameEncoder:
    """Accepts BGR uint8 frames and writes an H.264 mp4 (video only)."""

    def __init__(self, dst: Path, width: int, height: int, fps_str: str, crf: int = 18, preset: str = "slow"):
        self.dst = dst
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", fps_str, "-i", "-",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(dst)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        # communicate() closes stdin itself; closing it first makes communicate() raise on flush.
        _, err = self.proc.communicate()
        if self.proc.returncode != 0:
            raise FFmpegError("encoder failed: " + err.decode(errors="replace")[-2000:])

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def mux_audio(video: Path, original: Path, dst: Path, audio_codec: Optional[str],
              scale_to: Optional[tuple[int, int]] = None, crf: int = 18) -> None:
    """Combine enhanced video with the original's audio. Optionally rescale the video."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-i", original, "-map", "0:v:0"]
    if audio_codec:
        cmd += ["-map", "1:a:0?"]
    if scale_to:
        cmd += ["-vf", f"scale={scale_to[0]}:{scale_to[1]}:flags=lanczos",
                "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    if audio_codec:
        if audio_codec in COPYABLE_AUDIO:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-shortest", "-movflags", "+faststart", dst]
    run(cmd)


def validate_output(path: Path, expected_duration: float, tolerance: float = 0.03) -> VideoInfo:
    info = probe(path)
    if expected_duration > 0 and info.duration > 0:
        if abs(info.duration - expected_duration) > max(0.5, tolerance * expected_duration):
            raise FFmpegError(f"output duration {info.duration:.2f}s differs from source {expected_duration:.2f}s")
    return info
