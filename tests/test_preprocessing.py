import cv2
import numpy as np
import pytest

from processing import image_processing as ip
from processing.preprocessing import estimate_noise_sigma, preprocess_frame, resize_frame


def textured(h=240, w=320):
    """Well-lit, high-contrast, smooth test image (blurred checkerboard)."""
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // 30) + (xx // 30)) % 2 == 0, 70, 200).astype(np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(board, (0, 0), 3), cv2.COLOR_GRAY2BGR)


def test_resize_keeps_aspect_and_never_upscales():
    img = np.zeros((720, 1280, 3), np.uint8)
    out, scale = resize_frame(img, 640)
    assert out.shape == (360, 640, 3) and scale == 0.5
    small, s = resize_frame(np.zeros((100, 200, 3), np.uint8), 640)
    assert small.shape == (100, 200, 3) and s == 1.0


def test_rgb_conversion():
    img = np.zeros((10, 10, 3), np.uint8)
    img[..., 0] = 255                      # blue in BGR
    res = preprocess_frame(img, enhance=False)
    assert res.rgb[0, 0].tolist() == [0, 0, 255]


def test_dark_frame_triggers_clahe():
    dark = (textured() * 0.2).astype(np.uint8)
    res = preprocess_frame(dark)
    assert res.poor_lighting and "clahe" in res.applied
    assert cv2.cvtColor(res.bgr, cv2.COLOR_BGR2GRAY).std() > cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY).std()


def test_well_lit_frame_is_left_alone():
    res = preprocess_frame(textured())
    assert res.applied == []


def test_noise_estimate_and_denoise():
    clean = textured()
    noise = np.random.default_rng(1).normal(0, 15, clean.shape)
    noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)
    grey_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    grey_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    assert estimate_noise_sigma(grey_noisy) > estimate_noise_sigma(grey_clean) + 5
    assert "gaussian_denoise" in preprocess_frame(noisy).applied


def test_dark_noisy_frame_is_denoised_before_clahe():
    """Dark frames under-report noise; CLAHE would amplify it, so both must run."""
    dark = (textured() * 0.1).astype(np.uint8)
    noisy = np.clip(dark + np.random.default_rng(2).normal(0, 4, dark.shape), 0, 255).astype(np.uint8)
    assert preprocess_frame(noisy).applied == ["gaussian_denoise", "clahe"]


def test_empty_frame_raises():
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((0, 0, 3), np.uint8))


def test_sharpness_drops_with_blur():
    img = textured()
    img[::8] = 255                          # add sharp edges
    assert ip.sharpness(cv2.GaussianBlur(img, (15, 15), 0)) < ip.sharpness(img)


def test_binary_mask_morphology_removes_specks():
    prob = np.zeros((100, 100), np.float32)
    prob[20:80, 30:70] = 0.9                # the person
    prob[5, 5] = 0.9                        # isolated false positive
    mask = ip.binary_person_mask(prob)
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[5, 5] == 0 and mask[50, 50] == 255


def test_roi_from_landmarks():
    pts = np.array([[10, 10], [50, 90], [30, 40], [20, 60]], np.float32)
    vis = np.ones(4, np.float32)
    x1, y1, x2, y2 = ip.landmarks_roi(pts, vis, (100, 100, 3))
    assert x1 <= 10 and y1 <= 10 and x2 >= 50 and y2 >= 90
    assert ip.landmarks_roi(pts, np.zeros(4), (100, 100, 3)) is None
