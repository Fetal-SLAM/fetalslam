"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Serialized XFeat inference and homography matching for SLAM.
"""

import cv2
import numpy as np
import torch

import threading
from copy import deepcopy
from functools import wraps

from .modules.xfeat import XFeat
from ..runtime_config import runtime_settings

import logging as _logging

_logger = _logging.getLogger(__name__)


def synchronized_inference(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.inference_lock:
            return method(self, *args, **kwargs)
    return wrapper

class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        features = self.mtd.detectAndCompute(torch.tensor(x).permute(2,0,1).float()[None])[0]
        features['image_size'] = (x.shape[1], x.shape[0])
        return features

class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher

def init_method(method, max_kpts):
    if method == "ORB":
        return Method(descriptor=cv2.ORB_create(max_kpts, fastThreshold=10), matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True))
    elif method == "SIFT":
        return Method(descriptor=cv2.SIFT_create(max_kpts, contrastThreshold=-1, edgeThreshold=1000), matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True))
    elif method == "XFeat":
        model = XFeat(top_k=max_kpts)
        return Method(descriptor=CVWrapper(model), matcher=model)
    else:
        raise RuntimeError("Invalid Method.")


class Xfeat_Det_Matching:
    def __init__(self, max_kpts, img_margin=0.05, max_fake_features=200, settings=None):
        settings = settings or runtime_settings("features")
        self.settings = settings
        self.ref_frame = None
        self.current_frame = None
        self.inference_lock = threading.RLock()
        self.ref_precomp = [[],[]]
        self.H = None

        self.height = 720
        self.width = 720
        self.img_margin = int(self.height * img_margin)
        self.max_fake_features = max_fake_features


        #Homography params
        self.min_inliers = settings.min_inliers
        self.ransac_thr = settings.ransac_threshold
        self.shitomashi_features = None

        #Match params
        self.lightglue_min_confidence = settings.lightglue_min_confidence


        #Set local feature method here -- we expect cv2 or Kornia convention
        self.method = init_method("XFeat", max_kpts= max_kpts)
        from .modules.lighterglue import LighterGlue
        self.method.matcher.lighterglue = LighterGlue().eval()

    def matched_points(self, reference, current):
        """Match cached XFeat descriptors using LighterGlue, with no MNN fallback."""
        if len(reference['keypoints']) == 0 or len(current['keypoints']) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        for features in (reference, current):
            if 'image_size' not in features:
                raise ValueError('LighterGlue requires cached image_size in (width, height) order')
        idx0, idx1 = self.method.matcher.match_lighterglue(
            reference, current, min_conf=self.lightglue_min_confidence)
        points0 = reference['keypoints'][idx0].detach().cpu().numpy()
        points1 = current['keypoints'][idx1].detach().cpu().numpy()
        return points0, points1





    def warp_shitomasi_features(self, points, H, x_offset = 0):
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)

        warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
        warped_points_np[:, 0] += x_offset
        warped_shitomasi_points = warped_points_np.astype(int).reshape(-1,1,2)

        return warped_shitomasi_points




    @synchronized_inference
    def init_detect(self, image):
        if image is None:
            _logger.warning('%s', "[xFeat] Error: xFeat is not initialized! ref_frame is empty")
            return False, None, None
        else:
            self.ref_frame = image
            _logger.debug('%s', f"self.ref_frame is size: {self.ref_frame.shape}")
            self.ref_precomp = self.method.descriptor.detectAndCompute(image, None) #Cache ref features
            shitomashi_flag, shitomashi_features = self.detect_shitomasi_pyramid(image, max_features = self.max_fake_features)
            self.shitomashi_features = shitomashi_features.copy()
        return shitomashi_flag, shitomashi_features, self.ref_precomp


    #             break


    def detect_shitomasi_pyramid(self, image, start_quality=None, max_features=None, min_quality=None,
                              max_level=None, radius=None, mask=None, verbose=True):
        """
        Detect Shi-Tomasi corners with fallback to uniform grid-based sampling if not enough features.

        Args:
            image (np.ndarray): BGR input image.
            start_quality (float): Initial qualityLevel for Shi-Tomasi.
            max_features (int): Total desired number of features.
            min_quality (float): Minimum qualityLevel to reach.
            max_level (int): Number of quality-lowering pyramid levels.
            radius (int): Minimum distance between features.
            mask (np.ndarray or None): ROI mask (optional).
            verbose (bool): Emit detailed debug logs.

        Returns:
            success (bool): True if Shi-Tomasi found enough features, False if grid fallback used.
            keypoints (np.ndarray): (N, 1, 2) array of keypoints.
        """
        settings = self.settings
        start_quality = settings.corner_start_quality if start_quality is None else start_quality
        min_quality = settings.corner_min_quality if min_quality is None else min_quality
        max_features = self.max_fake_features if max_features is None else max_features
        max_level = settings.corner_levels if max_level is None else max_level
        radius = settings.corner_radius if radius is None else radius
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Create ROI mask if not provided
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[self.img_margin:h - self.img_margin,
                self.img_margin:w - self.img_margin] = 255

        features = []
        quality = start_quality

        for level in range(max_level):
            remaining = max_features - len(features)
            if remaining <= 0:
                break

            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=remaining,
                qualityLevel=quality,
                minDistance=radius,
                mask=mask,
                blockSize=settings.corner_block_size
            )

            if corners is not None:
                new_pts = [tuple(pt.ravel()) for pt in corners]
                features.extend(new_pts)

                # Suppress area to avoid overlap
                for x, y in new_pts:
                    cv2.circle(mask, (int(x), int(y)), radius=radius, color=0, thickness=-1)

            if verbose:
                _logger.debug('%s', f"[Level {level}] {len(features)} features, quality={quality:.4f}")

            if len(features) >= max_features:
                return True, np.array(features[:max_features], dtype=np.float32).reshape(-1, 1, 2)

            quality = max(min_quality, quality / settings.corner_quality_decay)

        needed = max_features - len(features)
        if needed > 0:
            if verbose:
                _logger.debug('%s', f"[Fallback] Generating {needed} uniform grid features")

            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                x0, y0, x1, y1 = self.img_margin, self.img_margin, w - self.img_margin, h - self.img_margin
            else:
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

            # Ensure grid_cols is at least 1 to avoid NaN
            grid_cols = max(1, int(np.ceil(np.sqrt(needed))))
            grid_rows = max(1, int(np.ceil(needed / grid_cols)))

            grid_x = np.linspace(x0, x1, grid_cols, endpoint=False)
            grid_y = np.linspace(y0, y1, grid_rows, endpoint=False)
            grid_pts = [(x, y) for y in grid_y for x in grid_x][:needed]

            features.extend(grid_pts)
        return False, np.array(features[:max_features], dtype=np.float32).reshape(-1, 1, 2)


    @synchronized_inference
    def detect(self, image):
        cur_precomp = self.method.descriptor.detectAndCompute(image, None)
        self.cur_precomp = cur_precomp

        return cur_precomp

    def estimate_homography(self, points1, points2):
        settings = getattr(self, "settings", None) or runtime_settings("features")
        if len(points1) < settings.homography_min_matches or len(points2) < settings.homography_min_matches:
            return None, None
        H, inliers = cv2.findHomography(
            points1,
            points2,
            cv2.USAC_MAGSAC,
            self.ransac_thr,
            maxIters=settings.ransac_iterations,
            confidence=settings.ransac_confidence,
        )
        if H is None or inliers is None:
            return None, None
        inlier_mask = inliers.reshape(-1) > 0
        if int(inlier_mask.sum()) < self.min_inliers:
            return None, None
        return H, inlier_mask


    @synchronized_inference
    def detect_and_match(self, image):
        if self.ref_frame is None:
            _logger.warning('%s', "[xFeat] Error: xFeat is not initialized! ref_frame is empty")
            return False, None, None
        self.current_frame = image.copy()

        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        cur_precomp = self.method.descriptor.detectAndCompute(self.current_frame)
        points1, points2 = self.matched_points(self.ref_precomp, cur_precomp)

        self.H, inliers = self.estimate_homography(points1, points2)
        if self.H is None:
            return False, None, None

        kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
        kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
        good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

        return True, self.H, good_matches


    @synchronized_inference
    def match(self, ref_precomp, cur_precomp):
        '''
        arg: input ref_precomp, cur_precomp
        return xfeat_flag, H_target_ref
        example usage
        input: kf_features in ref frame
        output: kf_features in target frame
        kf_features_in_target = self.xfeat.warp_shitomasi_features(kf_features, H_target_ref).reshape(-1,2).tolist()
        '''
        points1, points2 = self.matched_points(ref_precomp, cur_precomp)

        H, _ = self.estimate_homography(points1, points2)
        if H is None:
            return False, None
        return True, H


    @synchronized_inference
    def update_ref_frame(self, image):
        self.ref_frame = deepcopy(image)
        self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features
