import cv2
import numpy as np

from fetal_slam.loop_closing import LoopClosing
from fetal_slam.slam import evaluate_frame_quality


class AppearanceKeyFrame:
    def __init__(self, keyframe_id, frame_id, image):
        self.mnId = keyframe_id
        self.mnFrameId = frame_id
        self.image = image


def test_quality_gate_rejects_blank_blur_and_extreme_noise():
    blank = np.full((128, 128, 3), 127, dtype=np.uint8)
    checker = np.indices((128, 128)).sum(axis=0) // 8 % 2
    checker = cv2.cvtColor((checker * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(checker, (31, 31), 0)
    noise = np.random.default_rng(4).integers(0, 256, (128, 128, 3), dtype=np.uint8)

    blank_ok, _ = evaluate_frame_quality(blank, 6.0, 5000.0, 25.0)
    blur_ok, blur_quality = evaluate_frame_quality(blurred, 100.0, 5000.0, 25.0)
    noise_ok, noise_quality = evaluate_frame_quality(noise, 6.0, 5000.0, 25.0)

    assert not blank_ok
    assert not blur_ok and blur_quality["reason"] in {"blur", "low_contrast"}
    assert not noise_ok and noise_quality["reason"] == "noise"


def test_place_ranking_finds_old_visual_match_outside_temporal_window():
    closer = LoopClosing()
    closer.close_loop_frame_interval = 100
    rng = np.random.default_rng(7)
    old_place = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    unrelated = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(unrelated, (48, 48), 22, (255, 255, 255), -1)
    revisited = cv2.convertScaleAbs(old_place, alpha=0.95, beta=5)
    keyframes = {
        10: AppearanceKeyFrame(10, 10, old_place),
        20: AppearanceKeyFrame(20, 20, unrelated),
        950: AppearanceKeyFrame(950, 950, revisited),
    }

    ranked = closer.rank_place_candidates(revisited, keyframes, current_frame_id=1000)

    assert [keyframe_id for keyframe_id, _ in ranked] == [10, 20]
    assert ranked[0][1] > ranked[1][1]
