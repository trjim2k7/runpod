import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.engines.fast import upscale_frame


class NearestX2:
    """Stand-in for a spandrel ImageModelDescriptor: 2x nearest-neighbour upsample."""
    scale = 2
    supports_half = False

    def __call__(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def test_tiled_matches_untiled():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(93, 141, 3), dtype=np.uint8)  # odd dims, not tile-aligned
    whole = upscale_frame(NearestX2(), frame, tile=0, device="cpu")
    tiled = upscale_frame(NearestX2(), frame, tile=32, overlap=8, device="cpu")
    assert whole.shape == (186, 282, 3)
    assert np.array_equal(whole, tiled)
    # nearest x2 keeps the original pixels at even positions, in BGR order
    assert np.array_equal(whole[::2, ::2], frame)
