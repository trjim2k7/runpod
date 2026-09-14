import os

import pytest

from app.upload import UploadError, UploadManager, sanitize_filename


@pytest.fixture
def mgr(tmp_path):
    return UploadManager(upload_dir=tmp_path / "uploads", input_dir=tmp_path / "input", chunk_size=1000)


def _chunks(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]


def test_sanitize():
    assert sanitize_filename("My Video (1).MP4") == "My_Video_1.mp4"
    assert sanitize_filename("../../etc/passwd.mov") == "passwd.mov"
    with pytest.raises(UploadError):
        sanitize_filename("script.exe")


def test_roundtrip_out_of_order(mgr):
    data = os.urandom(2500)
    s = mgr.init("clip.mp4", len(data))
    assert s["total_chunks"] == 3
    parts = _chunks(data, 1000)
    for idx in (2, 0, 1):
        st = mgr.write_chunk(s["upload_id"], idx, [parts[idx][:400], parts[idx][400:]])
    assert st["received"] == [0, 1, 2]
    dst = mgr.complete(s["upload_id"])
    assert dst.read_bytes() == data
    assert dst.name == "clip.mp4"
    assert not (mgr.upload_dir / s["upload_id"]).exists()


def test_missing_chunk_rejected(mgr):
    data = os.urandom(2500)
    s = mgr.init("clip.mp4", len(data))
    mgr.write_chunk(s["upload_id"], 0, [data[:1000]])
    with pytest.raises(UploadError, match="missing"):
        mgr.complete(s["upload_id"])
    assert mgr.status(s["upload_id"])["received"] == [0]


def test_wrong_chunk_size_rejected(mgr):
    s = mgr.init("clip.mp4", 2500)
    with pytest.raises(UploadError):
        mgr.write_chunk(s["upload_id"], 0, [b"x" * 999])
    assert mgr.status(s["upload_id"])["received"] == []


def test_duplicate_names_get_suffix(mgr):
    for expected in ("a.mp4", "a_1.mp4", "a_2.mp4"):
        s = mgr.init("a.mp4", 10)
        mgr.write_chunk(s["upload_id"], 0, [b"0123456789"])
        assert mgr.complete(s["upload_id"]).name == expected


def test_bad_ids(mgr):
    with pytest.raises(UploadError):
        mgr.status("nope")
    with pytest.raises(UploadError):
        mgr.status("0" * 32)
