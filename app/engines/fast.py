"""Fast engine: Real-ESRGAN per frame (spandrel) + CodeFormer/GFPGAN face restoration (facexlib)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app import config
from app.ffmpeg import FrameDecoder, FrameEncoder, mux_audio, probe, target_resolution, validate_output

_MODEL_CACHE: dict = {}


def _load_upscaler(name: str):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    key = ("up", name)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    filename, _ = config.FAST_UPSCALERS[name]
    path = config.FAST_MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing weights {path}; run setup.sh")
    desc = ModelLoader().load_from_file(str(path))
    assert isinstance(desc, ImageModelDescriptor)
    desc = desc.cuda().eval()
    if desc.supports_half:
        desc = desc.half()
    _MODEL_CACHE[key] = desc
    return desc


def _load_face_model(name: str):
    import torch
    from spandrel import MAIN_REGISTRY, ModelLoader
    try:
        from spandrel_extra_arches import EXTRA_REGISTRY
        MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    except (ImportError, ValueError):
        pass  # ValueError: already registered

    key = ("face", name)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    filename, _ = config.FACE_MODELS[name]
    path = config.FAST_MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing weights {path}; run setup.sh")
    desc = ModelLoader().load_from_file(str(path)).cuda().eval()
    _MODEL_CACHE[key] = desc
    return desc


def _face_helper():
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    key = ("helper",)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
            use_parse=True, device="cuda", model_rootpath=str(config.FACEXLIB_MODELS_DIR))
    return _MODEL_CACHE[key]


def upscale_frame(desc, bgr: np.ndarray, tile: int = 0, overlap: int = 16, device: str = "cuda") -> np.ndarray:
    """Run a spandrel image model on a BGR uint8 frame, optionally tiled. Returns BGR uint8."""
    import torch

    dtype = torch.float16 if desc.supports_half else torch.float32
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    x = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)
    scale = desc.scale
    with torch.inference_mode():
        if tile <= 0 or (x.shape[2] <= tile and x.shape[3] <= tile):
            y = desc(x)
        else:
            _, _, h, w = x.shape
            y = torch.zeros((1, 3, h * scale, w * scale), dtype=dtype, device=x.device)
            for ty in range(0, h, tile):
                for tx in range(0, w, tile):
                    y0, y1 = ty, min(ty + tile, h)
                    x0, x1 = tx, min(tx + tile, w)
                    py0, py1 = max(0, y0 - overlap), min(h, y1 + overlap)
                    px0, px1 = max(0, x0 - overlap), min(w, x1 + overlap)
                    out = desc(x[:, :, py0:py1, px0:px1])
                    oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
                    y[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                        out[:, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
    # No in-place ops here: tensors made under inference_mode reject them once the block exits.
    out = (y[0].float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(out[:, :, ::-1])


def restore_faces(helper, face_desc, face_name: str, bgr: np.ndarray, fidelity: float) -> tuple:
    """Detect faces on a BGR frame, restore each 512x512 crop, paste back. Returns (frame, n_faces)."""
    import torch

    helper.clean_all()
    helper.read_image(bgr)
    n = helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    if not n:
        return bgr, 0
    helper.align_warp_face()
    for face in helper.cropped_faces:
        rgb = np.ascontiguousarray(face[:, :, ::-1])
        t = torch.from_numpy(rgb).cuda().permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        t = (t - 0.5) / 0.5  # both models were trained on [-1, 1] inputs
        with torch.inference_mode():
            try:
                if face_name == "CodeFormer":
                    out = face_desc.model(t, weight=float(fidelity))[0]
                else:
                    out = face_desc.model(t, return_rgb=False)[0]
            except RuntimeError:
                out = t  # keep the unrestored crop rather than fail the whole video
        out = ((out[0].float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        helper.add_restored_face(np.ascontiguousarray(out[:, :, ::-1]))
    helper.get_inverse_affine(None)
    pasted = helper.paste_faces_to_input_image(upsample_img=bgr)
    return pasted, len(helper.cropped_faces)


class FastEngine:
    name = "fast"

    def run(self, job, on_progress, cancel_event: threading.Event) -> Path:
        import cv2

        src = Path(job.input_path)
        opts = job.options
        info = probe(src)
        tw, th = target_resolution(info.width, info.height)
        job.log_line(f"source {info.width}x{info.height} @ {float(info.fps):.3f} fps, {info.duration:.1f}s, "
                     f"audio={info.audio_codec}; target {tw}x{th}")

        on_progress(0.0, "loading models")
        up_name = opts.get("upscaler", "RealESRGAN_x4plus")
        upscaler = _load_upscaler(up_name)
        face_on = bool(opts.get("face_restore", True))
        face_name = opts.get("face_model", "CodeFormer")
        fidelity = float(opts.get("fidelity", 0.6))
        face_desc = _load_face_model(face_name) if face_on else None
        helper = _face_helper() if face_on else None
        tile = int(opts.get("tile", 0))
        if tile == 0 and info.width * info.height > 1280 * 720:
            tile = 512
        job.log_line(f"upscaler={up_name} x{upscaler.scale} tile={tile} face_restore={face_on} "
                     f"face_model={face_name if face_on else '-'} fidelity={fidelity}")

        work = job.work_dir
        noaudio = work / "video_noaudio.mp4"
        total = max(1, info.nb_frames)
        done = 0
        faces_total = 0
        encoder = FrameEncoder(noaudio, tw, th, info.fps_str, crf=int(opts.get("crf", 18)))
        try:
            with FrameDecoder(src, info.width, info.height, info.fps_str) as dec:
                for frame in dec.frames():
                    if cancel_event.is_set():
                        raise RuntimeError("cancelled")
                    up = upscale_frame(upscaler, frame, tile=tile)
                    if (up.shape[1], up.shape[0]) != (tw, th):
                        interp = cv2.INTER_AREA if up.shape[1] > tw else cv2.INTER_LANCZOS4
                        up = cv2.resize(up, (tw, th), interpolation=interp)
                    if face_on:
                        up, n = restore_faces(helper, face_desc, face_name, up, fidelity)
                        faces_total += n
                    encoder.write(up)
                    done += 1
                    if done % 10 == 0 or done == total:
                        on_progress(0.02 + 0.93 * min(1.0, done / total), f"frame {done}/{total}")
                    if done % 200 == 0:
                        job.log_line(f"frame {done}/{total}, faces so far {faces_total}")
            encoder.close()
        except BaseException:
            encoder.abort()
            raise
        job.log_line(f"encoded {done} frames, restored {faces_total} face crops")

        on_progress(0.96, "muxing audio")
        final = work / "output.mp4"
        mux_audio(noaudio, src, final, info.audio_codec, crf=int(opts.get("crf", 18)))
        validate_output(final, info.duration)
        on_progress(1.0, "done")
        return final
