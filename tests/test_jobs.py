import threading
import time
from pathlib import Path

import pytest

from app import jobs as jobs_mod
from app.jobs import CANCELLED, DONE, FAILED, JobManager


class DummyEngine:
    name = "dummy"

    def __init__(self, fail=False, slow=False):
        self.fail, self.slow = fail, slow

    def run(self, job, on_progress, cancel_event):
        on_progress(0.5, "halfway")
        if self.fail:
            raise RuntimeError("boom")
        for _ in range(50 if self.slow else 1):
            if cancel_event.is_set():
                raise RuntimeError("cancelled")
            time.sleep(0.05)
        out = job.work_dir / "output.mp4"
        out.write_bytes(b"video")
        return out


def _wait(job, timeout=5.0):
    t0 = time.time()
    while job.state not in jobs_mod.TERMINAL and time.time() - t0 < timeout:
        time.sleep(0.02)
    return job.state


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    engines = {}
    monkeypatch.setattr("app.engines.get_engine", lambda name: engines[name])
    m = JobManager(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "out")
    m._engines = engines
    return m


def _src(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"x")
    return p


def test_success_publishes_output(mgr, tmp_path):
    mgr._engines["ok"] = DummyEngine()
    job = mgr.submit(_src(tmp_path), "ok", {"a": 1})
    assert _wait(job) == DONE
    assert job.progress == 1.0
    out = Path(job.output_path)
    assert out.exists() and out.name.startswith("clip_ok_1080p_") and out.read_bytes() == b"video"
    for _ in range(100):  # work dir is removed just after the state flips to DONE
        if not job.work_dir.exists():
            break
        time.sleep(0.02)
    assert not job.work_dir.exists()
    assert (job.dir / "job.json").exists()


def test_failure_recorded(mgr, tmp_path):
    mgr._engines["bad"] = DummyEngine(fail=True)
    job = mgr.submit(_src(tmp_path), "bad", {})
    assert _wait(job) == FAILED
    assert "boom" in job.error
    assert any("Traceback" in line for line in job.log)


def test_cancel_running(mgr, tmp_path):
    mgr._engines["slow"] = DummyEngine(slow=True)
    job = mgr.submit(_src(tmp_path), "slow", {})
    time.sleep(0.3)
    assert job.state == jobs_mod.RUNNING
    assert mgr.cancel(job.id)
    assert _wait(job) == CANCELLED
    assert job.output_path is None


def test_history_reload_marks_interrupted(tmp_path):
    m = JobManager(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "out", start_worker=False)
    job = m.submit(tmp_path / "a.mp4", "x", {})
    job.state = jobs_mod.RUNNING
    m._save(job)
    m2 = JobManager(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "out", start_worker=False)
    assert m2.get(job.id).state == FAILED
    assert "restarted" in m2.get(job.id).error
