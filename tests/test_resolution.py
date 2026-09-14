from app.ffmpeg import target_resolution


def test_landscape_480p_to_1080p():
    assert target_resolution(640, 480) == (1440, 1080)


def test_landscape_720p():
    assert target_resolution(1280, 720) == (1920, 1080)


def test_portrait_phone():
    assert target_resolution(480, 848) == (1080, 1908)


def test_never_downscale_by_default():
    assert target_resolution(3840, 2160) == (3840, 2160)
    assert target_resolution(3840, 2160, allow_downscale=True) == (1920, 1080)


def test_already_1080p():
    assert target_resolution(1920, 1080) == (1920, 1080)


def test_even_dimensions():
    w, h = target_resolution(427, 240)
    assert w % 2 == 0 and h % 2 == 0 and w == 1920


def test_wide_aspect_limited_by_long_side():
    assert target_resolution(1000, 300) == (1920, 576)
