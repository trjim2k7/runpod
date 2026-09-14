"""Single-worker job queue. Jobs persist as jobs/<id>/job.json so history survives restarts."""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app import config

log = logging.getLogger("enhance.jobs")

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"
TERMINAL = {DONE, FAILED, CANCELLED}


@dataclass
class Job:
    id: str
    input_path: str
    engine: str
    options: dict
    dir: Path
    state: str = QUEUED
    progress: float = 0.0
    stage: str = "queued"
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    log: deque = field(default_factory=lambda: deque(maxlen=400))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    proc: Any = None  # subprocess handle when an engine runs a child process

    @property
    def work_dir(self) -> Path:
        return self.dir / "work"

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    @property
    def eta_s(self) -> Optional[float]:
        if self.state != RUNNING or self.progress <= 0.02:
            return None
        return self.elapsed * (1 - self.progress) / self.progress

    def log_line(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        self.log.append(line)
        try:
            with open(self.dir / "log.txt", "a", encoding="utf-8", errors="replace") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def to_dict(self) -> dict:
        return {
            "id": self.id, "input_path": self.input_path, "engine": self.engine,
            "options": self.options, "state": self.state, "progress": self.progress,
            "stage": self.stage, "output_path": self.output_path, "error": self.error,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, d: dict, jobs_dir: Path) -> "Job":
        j = cls(id=d["id"], input_path=d["input_path"], engine=d["engine"],
                options=d.get("options", {}), dir=jobs_dir / d["id"])
        for k in ("state", "progress", "stage", "output_path", "error",
                  "created_at", "started_at", "finished_at"):
            if k in d:
                setattr(j, k, d[k])
        try:
            lines = (j.dir / "log.txt").read_text(encoding="utf-8", errors="replace").splitlines()
            j.log.extend(lines[-100:])
        except OSError:
            pass
        return j


class JobManager:
    def __init__(self, jobs_dir: Path = config.JOBS_DIR, output_dir: Path = config.OUTPUT_DIR,
                 start_worker: bool = True):
        self.jobs_dir = jobs_dir
        self.output_dir = output_dir
        self.jobs: dict = {}
        self.queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.current: Optional[str] = None
        self._load_history()
        if start_worker:
            self.worker = threading.Thread(target=self._loop, name="job-worker", daemon=True)
            self.worker.start()

    # ---- persistence --------------------------------------------------
    def _load_history(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        for jf in sorted(self.jobs_dir.glob("*/job.json")):
            try:
                job = Job.from_dict(json.loads(jf.read_text(encoding="utf-8")), self.jobs_dir)
            except (OSError, ValueError, KeyError) as e:
                log.warning("skipping %s: %s", jf, e)
                continue
            if job.state in (RUNNING, QUEUED):
                job.state, job.error = FAILED, "server restarted while job was active"
                job.finished_at = job.finished_at or time.time()
                self._save(job)
                shutil.rmtree(job.work_dir, ignore_errors=True)
            self.jobs[job.id] = job

    def _save(self, job: Job) -> None:
        job.dir.mkdir(parents=True, exist_ok=True)
        tmp = job.dir / "job.json.tmp"
        tmp.write_text(json.dumps(job.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, job.dir / "job.json")

    # ---- public API ---------------------------------------------------
    def submit(self, input_path: Path, engine: str, options: dict) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, input_path=str(input_path), engine=engine, options=options,
                  dir=self.jobs_dir / job_id)
        with self.lock:
            self.jobs[job_id] = job
            self._save(job)
        job.log_line(f"queued {Path(input_path).name} with engine={engine} options={options}")
        self.queue.put(job_id)
        return job

    def get(self, job_id: Optional[str]) -> Optional[Job]:
        return self.jobs.get(job_id) if job_id else None

    def list(self) -> list:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.state in TERMINAL:
            return False
        job.cancel_event.set()
        job.log_line("cancel requested")
        if job.state == QUEUED:
            self._finish(job, CANCELLED, "cancelled before start")
        return True

    def delete(self, job_id: str, delete_output: bool = False) -> None:
        job = self.jobs.get(job_id)
        if not job or job.state not in TERMINAL:
            return
        with self.lock:
            self.jobs.pop(job_id, None)
        shutil.rmtree(job.dir, ignore_errors=True)
        if delete_output and job.output_path:
            Path(job.output_path).unlink(missing_ok=True)

    def sweep(self, max_age_days: float = config.JOB_MAX_AGE_DAYS) -> int:
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for job in list(self.jobs.values()):
            if job.state in TERMINAL and (job.finished_at or job.created_at) < cutoff:
                self.delete(job.id)
                removed += 1
        return removed

    # ---- worker -------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.state != QUEUED:
                continue
            self.current = job_id
            try:
                self._run(job)
            finally:
                self.current = None

    def _run(self, job: Job) -> None:
        from app.engines import get_engine

        job.state, job.stage, job.started_at = RUNNING, "starting", time.time()
        self._save(job)
        job.work_dir.mkdir(parents=True, exist_ok=True)

        def on_progress(fraction: float, stage: str) -> None:
            job.progress = max(0.0, min(1.0, fraction))
            job.stage = stage

        try:
            engine = get_engine(job.engine)
            out = engine.run(job, on_progress, job.cancel_event)
            if job.cancel_event.is_set():
                self._finish(job, CANCELLED, "cancelled")
                return
            final = self._publish(job, Path(out))
            job.output_path = str(final)
            job.progress = 1.0
            self._finish(job, DONE)
        except Exception as e:  # noqa: BLE001 - report anything the engine raises
            if job.cancel_event.is_set():
                self._finish(job, CANCELLED, "cancelled")
            else:
                job.log_line(traceback.format_exc())
                self._finish(job, FAILED, f"{type(e).__name__}: {e}")
        finally:
            shutil.rmtree(job.work_dir, ignore_errors=True)
            self._free_gpu()

    def _publish(self, job: Job, out: Path) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(job.input_path).stem
        final = self.output_dir / f"{stem}_{job.engine}_1080p_{job.id[:8]}.mp4"
        shutil.move(str(out), str(final))
        return final

    def _finish(self, job: Job, state: str, error: Optional[str] = None) -> None:
        job.state = state
        job.error = error
        job.finished_at = time.time()
        job.stage = state.lower()
        job.log_line(state + (f": {error}" if error else ""))
        self._save(job)

    @staticmethod
    def _free_gpu() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
