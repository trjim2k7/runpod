"""Chunked upload sessions: init -> N x chunk -> complete. Parts live in uploads/<id>/ until assembled."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Iterable, Optional

from app import config

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class UploadError(ValueError):
    pass


def sanitize_filename(name: str) -> str:
    name = os.path.basename(name.replace("\\", "/"))
    stem, ext = os.path.splitext(name)
    ext = ext.lower()
    if ext not in config.ALLOWED_EXT:
        raise UploadError(f"file type {ext or '(none)'} not allowed")
    stem = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "video"
    return f"{stem[:80]}{ext}"


def unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    stem, ext = os.path.splitext(filename)
    n = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{ext}"
        n += 1
    return candidate


class UploadManager:
    def __init__(self, upload_dir: Path = config.UPLOAD_DIR, input_dir: Path = config.INPUT_DIR,
                 chunk_size: int = config.CHUNK_SIZE):
        self.upload_dir = upload_dir
        self.input_dir = input_dir
        self.chunk_size = chunk_size

    # ---- internals ----------------------------------------------------
    def _dir(self, upload_id: str) -> Path:
        if not _ID_RE.match(upload_id or ""):
            raise UploadError("bad upload id")
        d = self.upload_dir / upload_id
        if not d.is_dir():
            raise UploadError("unknown upload id")
        return d

    def _meta(self, d: Path) -> dict:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))

    @staticmethod
    def _part(d: Path, index: int) -> Path:
        return d / f"part_{index:05d}"

    # ---- protocol -----------------------------------------------------
    def init(self, filename: str, size: int) -> dict:
        if size <= 0:
            raise UploadError("empty file")
        safe = sanitize_filename(filename)
        upload_id = uuid.uuid4().hex
        d = self.upload_dir / upload_id
        d.mkdir(parents=True, exist_ok=False)
        total = (size + self.chunk_size - 1) // self.chunk_size
        meta = {"filename": safe, "size": size, "chunk_size": self.chunk_size,
                "total_chunks": total, "created_at": time.time()}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return {"upload_id": upload_id, "chunk_size": self.chunk_size, "total_chunks": total}

    def write_chunk(self, upload_id: str, index: int, chunks: Iterable[bytes]) -> dict:
        d = self._dir(upload_id)
        meta = self._meta(d)
        if not 0 <= index < meta["total_chunks"]:
            raise UploadError("chunk index out of range")
        expected = min(meta["chunk_size"], meta["size"] - index * meta["chunk_size"])
        tmp = self._part(d, index).with_suffix(".tmp")
        written = 0
        with open(tmp, "wb") as f:
            for piece in chunks:
                f.write(piece)
                written += len(piece)
                if written > expected:
                    break
        if written != expected:
            tmp.unlink(missing_ok=True)
            raise UploadError(f"chunk {index}: got {written} bytes, expected {expected}")
        os.replace(tmp, self._part(d, index))
        return self.status(upload_id)

    def status(self, upload_id: str) -> dict:
        d = self._dir(upload_id)
        meta = self._meta(d)
        received = sorted(int(p.name[5:]) for p in d.glob("part_[0-9]*") if p.suffix != ".tmp")
        return {"upload_id": upload_id, "received": received, "total_chunks": meta["total_chunks"],
                "chunk_size": meta["chunk_size"], "filename": meta["filename"]}

    def complete(self, upload_id: str) -> Path:
        d = self._dir(upload_id)
        meta = self._meta(d)
        missing = [i for i in range(meta["total_chunks"]) if not self._part(d, i).exists()]
        if missing:
            raise UploadError(f"missing chunks: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        self.input_dir.mkdir(parents=True, exist_ok=True)
        dst = unique_path(self.input_dir, meta["filename"])
        tmp = dst.with_name(dst.name + ".assembling")
        total = 0
        with open(tmp, "wb") as out:
            for i in range(meta["total_chunks"]):
                with open(self._part(d, i), "rb") as part:
                    total += shutil.copyfileobj(part, out) or self._part(d, i).stat().st_size
        if tmp.stat().st_size != meta["size"]:
            tmp.unlink(missing_ok=True)
            raise UploadError(f"assembled size {tmp.stat().st_size if tmp.exists() else 0} != {meta['size']}")
        os.replace(tmp, dst)
        shutil.rmtree(d, ignore_errors=True)
        return dst

    def abort(self, upload_id: str) -> None:
        shutil.rmtree(self._dir(upload_id), ignore_errors=True)

    def sweep(self, max_age_h: float = config.UPLOAD_MAX_AGE_H) -> int:
        if not self.upload_dir.is_dir():
            return 0
        cutoff = time.time() - max_age_h * 3600
        n = 0
        for d in self.upload_dir.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    n += 1
            except OSError:
                pass
        return n


def list_inputs(input_dir: Path = config.INPUT_DIR) -> list[Path]:
    if not input_dir.is_dir():
        return []
    files = [p for p in input_dir.iterdir()
             if p.is_file() and p.suffix.lower() in config.ALLOWED_EXT and not p.name.endswith(".assembling")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
