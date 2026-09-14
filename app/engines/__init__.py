"""Engine registry. Each engine exposes run(job, on_progress, cancel_event) -> Path of output mp4."""
from __future__ import annotations

from typing import Callable, Protocol, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from pathlib import Path
    from app.jobs import Job

ProgressFn = Callable[[float, str], None]


class Engine(Protocol):
    name: str

    def run(self, job: "Job", on_progress: ProgressFn, cancel_event: threading.Event) -> "Path": ...


def get_engine(name: str) -> Engine:
    if name == "seedvr2":
        from app.engines.seedvr2 import SeedVR2Engine
        return SeedVR2Engine()
    if name == "fast":
        from app.engines.fast import FastEngine
        return FastEngine()
    raise ValueError(f"unknown engine: {name}")


ENGINE_LABELS = {
    "seedvr2": "Quality (SeedVR2)",
    "fast": "Fast (Real-ESRGAN + face restore)",
}
