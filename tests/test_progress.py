import pytest

from app.engines.seedvr2 import parse_progress


@pytest.mark.parametrize("line,expected", [
    ("Written 50/200 frames", 0.9625),
    ("  Written 200 / 200 frames", 1.0),
    ("Chunk 3/10: 197 new + 3 context frames", 0.2),
    ("Chunk 1/4: 200 new frames", 0.0),
    ("Upscaling:  45%|####      | 9/20 [00:10<00:12,  1.1s/it]", 0.45),
    ("9/20 [00:10<00:12, 1.1it/s]", 0.45),
    ("[19:43:11.080] 🎬 Upscaling batch 1/165", 0.0),
    ("[19:43:11.080] 🎬 Upscaling batch 80/165", 79 / 165 * 0.475),
    ("[19:45:10.544] 🎨 Decoding batch 25/165", 0.475 + 24 / 165 * 0.475),
    ("[19:45:10.545] 🎨   Using VAE tiled decoding (Tile: (1024, 1024), Overlap: (128, 128))", None),
    ("EulerSampler: 100%|██████████| 1/1 [00:00<00:00,  1.32it/s]", None),
    ("EulerSampler:   0%|          | 0/1 [00:00<?, ?it/s]", None),
    ("Loading model seedvr2_ema_3b_fp16.safetensors", None),
    ("Using 16/32 GB VRAM", None),
    ("", None),
])
def test_parse_progress(line, expected):
    got = parse_progress(line)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)
