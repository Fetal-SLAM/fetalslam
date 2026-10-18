import numpy as np
import cv2

from .features.detector import Xfeat_Det_Matching
from .optimizer import Optimizer
from .visual_reconstruction import VisualReconstruction
from .keyframe import KeyFrameTree
from .map import Mapping

import logging as _logging

_logger = _logging.getLogger(__name__)

from .runtime_config import runtime_settings


class LoopClosing():
    def __init__(self, width = 720, height = 720, margin = 0.1, close_loop_iou_threshold=0.65, settings=None):
        self.settings = settings or runtime_settings("loop_closing")
        self.height = height
        self.width = width
        self.margin = int(margin * self.width)

        self.close_loop_iou_threshold = close_loop_iou_threshold

        self.close_loop_frame_interval = 120
        self.close_loop_cooldown_frames = 150
        self.place_candidate_count = 12
        self.loop_confirmation_count = 2
        self.relocalization_iou_threshold = 0.45
        self.pending_loop_candidate_frame_id = None
        self.pending_loop_candidate_count = 0
        self.last_loop_query_frame_id = None
        self.appearance_descriptor_cache = {}

        self.local_map_recovery_enabled = False
        self.local_map_recovery_interval = 3
        self.local_map_recovery_min_frame_gap = 30
        self.local_map_recovery_candidate_count = 4
        self.local_map_recovery_iou_threshold = 0.30
        self.local_map_recovery_min_inliers = 15
        self.local_map_recovery_min_observations = 2
        self.local_map_recovery_max_points = 120
        self.local_map_recovery_reprojection_error = 4.0
        self.local_map_recovery_max_pose_jump_depth_factor = 0.1
        self.local_map_recovery_jobs = 0
        
        self.covisible_iou_threshold = self.settings.covisible_iou_threshold

        self.degub = False

        self.last_close_loop_keyframe_id = 0

    def set_xfeat(self, xfeat:Xfeat_Det_Matching):
        self.xfeat = xfeat

    def set_optimizer(self, optimizer:Optimizer):
        self.optimizer = optimizer

    def set_vis_recon(self, vis_con:VisualReconstruction):
        self.vis_con = vis_con #Class for visual reconstruction

    def set_map(self, global_map:Mapping):
        self.map = global_map

    def set_KeyFrameTree(self, KFtree:KeyFrameTree):
        self.KFtree = KFtree

    def configure_place_recognition(
        self,
        min_frame_gap,
        cooldown_frames,
        candidate_count,
        confirmation_count,
        relocalization_iou_threshold,
    ):
        self.close_loop_frame_interval = int(min_frame_gap)
        self.close_loop_cooldown_frames = int(cooldown_frames)
        self.place_candidate_count = int(candidate_count)
        self.loop_confirmation_count = int(confirmation_count)
        self.relocalization_iou_threshold = float(relocalization_iou_threshold)

    def configure_local_map_recovery(
        self,
        enabled,
        interval,
        min_frame_gap,
        candidate_count,
        iou_threshold,
        min_inliers,
        min_observations,
        max_points,
        reprojection_error,
        max_pose_jump_depth_factor,
    ):
        self.local_map_recovery_enabled = bool(enabled)
        self.local_map_recovery_interval = max(1, int(interval))
        self.local_map_recovery_min_frame_gap = max(1, int(min_frame_gap))
        self.local_map_recovery_candidate_count = max(1, int(candidate_count))
        self.local_map_recovery_iou_threshold = float(iou_threshold)
        self.local_map_recovery_min_inliers = max(4, int(min_inliers))
        self.local_map_recovery_min_observations = max(1, int(min_observations))
        self.local_map_recovery_max_points = max(4, int(max_points))
        self.local_map_recovery_reprojection_error = float(reprojection_error)
        self.local_map_recovery_max_pose_jump_depth_factor = float(max_pose_jump_depth_factor)

    @staticmethod
    def compute_appearance_descriptor(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        normalized = cv2.equalizeHist(small).astype(np.float32) / 255.0
        dct = cv2.dct(normalized)[:8, :8].reshape(-1)
        dct[0] = 0.0
        dct /= max(float(np.linalg.norm(dct)), 1e-6)

        histogram = cv2.calcHist([small], [0], None, [16], [0, 256]).reshape(-1)
        histogram /= max(float(np.linalg.norm(histogram)), 1e-6)
        return np.concatenate((dct, histogram)).astype(np.float32) / np.sqrt(2.0)

    def rank_place_candidates(
        self, image, keyframes, current_frame_id, limit=None, min_frame_gap=None
    ):
        if image is None:
            return []
        limit = self.place_candidate_count if limit is None else int(limit)
        min_frame_gap = (
            self.close_loop_frame_interval
            if min_frame_gap is None
            else int(min_frame_gap)
        )
        query_descriptor = self.compute_appearance_descriptor(image)
        ranked = []
        for keyframe_id, keyframe in keyframes.items():
            if abs(current_frame_id - keyframe.mnFrameId) < min_frame_gap:
                continue
            descriptor = self.appearance_descriptor_cache.get(keyframe_id)
            if descriptor is None:
                descriptor = self.compute_appearance_descriptor(keyframe.image)
                self.appearance_descriptor_cache[keyframe_id] = descriptor
            ranked.append((keyframe_id, float(np.dot(query_descriptor, descriptor))))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    @staticmethod
    def _local_recovery_result(reason, attempted=False, **values):
        result = {
            "attempted": attempted,
            "accepted": False,
            "candidate_id": None,
            "iou": 0.0,
            "inliers": 0,
            "added_points": 0,
            "reason": reason,
        }
        result.update(values)
        return result

    def _build_recovery_correspondences(self, reference, current, homography):
        reference_features = reference.get_KeysUn_dict()
        reference_points = reference.get_mvpMapPoints()
        current_features = current.get_KeysUn_dict()
        feature_ids = [
            feature_id
            for feature_id in reference_features
            if feature_id not in current_features
            and feature_id in reference_points
            and reference_points[feature_id] is not None
            and not reference_points[feature_id].is_bad()
            and reference_points[feature_id].Observations()
            >= self.local_map_recovery_min_observations
        ]
        if not feature_ids:
            return []

        source_features = [reference_features[feature_id] for feature_id in feature_ids]
        warped = self.xfeat.warp_shitomasi_features(source_features, homography)
        warped = np.asarray(warped, dtype=np.float64).reshape(-1, 2)
        existing = np.asarray(list(current_features.values()), dtype=np.float64).reshape(-1, 2)

        candidates = []
        for feature_id, feature in zip(feature_ids, warped):
            if not np.all(np.isfinite(feature)):
                continue
            x, y = feature
            if not (2.0 <= x < self.width - 2.0 and 2.0 <= y < self.height - 2.0):
                continue
            if len(existing) and np.min(np.linalg.norm(existing - feature, axis=1)) < 3.0:
                continue
            candidates.append((feature_id, reference_points[feature_id], feature))

        if len(candidates) > self.local_map_recovery_max_points:
            selected = np.linspace(
                0, len(candidates) - 1, self.local_map_recovery_max_points, dtype=int
            )
            candidates = [candidates[index] for index in selected]
        return candidates

    def _solve_recovery_pose(self, current, correspondences):
        if len(correspondences) < self.local_map_recovery_min_inliers:
            return None, np.empty(0, dtype=int)

        combined = list(correspondences)
        current_features = current.get_KeysUn_dict()
        for point_id, point in current.get_mvpMapPoints().items():
            feature = current_features.get(point_id)
            if point is not None and feature is not None and not point.is_bad():
                if np.all(np.isfinite(point.get_world_pos())) and np.all(np.isfinite(feature)):
                    combined.append((point_id, point, feature))
        world_points = np.asarray(
            [item[1].get_world_pos().reshape(3) for item in combined],
            dtype=np.float64,
        )
        image_points = np.asarray([item[2] for item in combined], dtype=np.float64)
        initial_pose = current.get_pose()
        rvec, _ = cv2.Rodrigues(initial_pose[:3, :3])
        tvec = initial_pose[:3, 3].reshape(3, 1).copy()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_points,
            image_points,
            self.optimizer.mK.astype(np.float64),
            None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=self.settings.pnp_iterations,
            reprojectionError=self.local_map_recovery_reprojection_error,
            confidence=self.settings.pnp_confidence,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None:
            return None, np.empty(0, dtype=int)
        inliers = inliers.reshape(-1)
        support_count = len(combined) - len(correspondences)
        if support_count >= 6 and np.count_nonzero(inliers >= len(correspondences)) < support_count * self.settings.pnp_support_fraction:
            return None, np.empty(0, dtype=int)
        inliers = inliers[inliers < len(correspondences)]
        if (
            len(inliers) < self.local_map_recovery_min_inliers
            or len(inliers) / len(correspondences) < self.settings.pnp_inlier_fraction
        ):
            return None, inliers

        rotation, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = tvec.reshape(3)
        return pose, inliers

    def _recovery_pose_is_plausible(self, current, old_pose, recovered_pose):
        if not np.all(np.isfinite(recovered_pose)):
            return False
        old_center = -old_pose[:3, :3].T @ old_pose[:3, 3]
        new_center = -recovered_pose[:3, :3].T @ recovered_pose[:3, 3]
        jump = float(np.linalg.norm(new_center - old_center))
        median_depth = float(current.compute_scene_median_depth(2))
        if median_depth <= 0.0:
            return True
        return jump <= self.local_map_recovery_max_pose_jump_depth_factor * median_depth

    def recover_local_map(self, parent_name):
        """Recover persistent map-point observations into a recent parent keyframe."""
        if not self.local_map_recovery_enabled:
            return self._local_recovery_result("disabled")
        if parent_name not in self.KFtree.parents:
            return self._local_recovery_result("parent-not-found")

        self.local_map_recovery_jobs += 1
        if (self.local_map_recovery_jobs - 1) % self.local_map_recovery_interval:
            return self._local_recovery_result("interval")

        current = self.KFtree.parents[parent_name]["node"].get_kf()
        grandparents = {
            name: node.get_kf()
            for name, node in self.KFtree.grand_parents.items()
            if node.get_kf().mnId != current.mnId
        }
        ranked = self.rank_place_candidates(
            current.image,
            grandparents,
            current.mnFrameId,
            limit=self.local_map_recovery_candidate_count,
            min_frame_gap=self.local_map_recovery_min_frame_gap,
        )
        if not ranked:
            return self._local_recovery_result("no-candidates", attempted=True)

        current_xfeat = current.get_xfeat_features()
        if current_xfeat is None:
            return self._local_recovery_result("current-xfeat-missing", attempted=True)

        best = None
        for candidate_name, _ in ranked:
            candidate = grandparents[candidate_name]
            candidate_xfeat = candidate.get_xfeat_features()
            if candidate_xfeat is None:
                continue
            matched, homography = self.xfeat.match(candidate_xfeat, current_xfeat)
            if not matched or homography is None or not np.all(np.isfinite(homography)):
                continue
            iou = float(self.vis_con.compute_iou_from_homography_fast(homography))
            if iou < self.local_map_recovery_iou_threshold:
                continue
            correspondences = self._build_recovery_correspondences(
                candidate, current, homography
            )
            pose, inliers = self._solve_recovery_pose(current, correspondences)
            if pose is None:
                continue
            inliers = self._filter_recovery_observations(current, correspondences, inliers, pose)
            if len(inliers) < self.local_map_recovery_min_inliers:
                continue
            score = (len(inliers), iou)
            if best is None or score > best[0]:
                best = (score, candidate_name, candidate, correspondences, pose, inliers)

        if best is None:
            return self._local_recovery_result("geometric-verification-failed", attempted=True)

        _, candidate_name, candidate, correspondences, recovered_pose, inliers = best
        old_pose = current.get_pose()
        if not self._recovery_pose_is_plausible(current, old_pose, recovered_pose):
            return self._local_recovery_result(
                "pose-jump", attempted=True, candidate_id=candidate_name,
                iou=best[0][1], inliers=len(inliers)
            )

        added = []
        try:
            current.set_pose(recovered_pose)
            for index in inliers:
                feature_id, map_point, feature = correspondences[int(index)]
                if current.update_map_point(map_point, feature, feature_id):
                    had_observation = current.mnId in map_point.get_observations()
                    added.append((feature_id, map_point, had_observation))
                    map_point.add_observation(current.mnId)
            if len(added) < self.local_map_recovery_min_inliers:
                raise RuntimeError("too few observations inserted")
        except Exception as error:
            current.set_pose(old_pose)
            for feature_id, map_point, had_observation in added:
                current.delete_map_point_match(feature_id)
                if not had_observation:
                    map_point.erase_observation(current.mnId)
            _logger.warning('%s', f"local map recovery rolled back: {error}")
            return self._local_recovery_result(
                "insertion-failed", attempted=True, candidate_id=candidate_name,
                iou=best[0][1], inliers=len(inliers)
            )

        for _, map_point, _ in added:
            map_point.local_recovery_protected = True

        return self._local_recovery_result(
            "accepted",
            attempted=True,
            accepted=True,
            candidate_id=candidate_name,
            iou=best[0][1],
            inliers=len(inliers),
            added_points=len(added),
        )

    def _filter_recovery_observations(self, current, correspondences, inliers, pose=None):
        """Verify observations against fixed map geometry and a candidate pose."""
        pose = current.get_pose() if pose is None else pose
        occupied = list(current.get_KeysUn_dict().values())
        accepted = []
        for index in inliers:
            _, point, feature = correspondences[int(index)]
            camera_point = pose[:3, :3] @ point.get_world_pos().reshape(3) + pose[:3, 3]
            if not np.all(np.isfinite(camera_point)) or camera_point[2] <= 1e-8:
                continue
            projected = self.optimizer.mK @ camera_point
            projected = projected[:2] / projected[2]
            if np.linalg.norm(projected - feature) > self.local_map_recovery_reprojection_error:
                continue
            if occupied and np.min(np.linalg.norm(np.asarray(occupied).reshape(-1, 2) - feature, axis=1)) < 3.0:
                continue
            accepted.append(int(index))
            occupied.append(feature)
        return np.asarray(accepted, dtype=int)


    def run(self, force=False, query_keyframe=None):
        
        if self.degub:
            return None, None, None
        
        # start with a new parent is reconstructed
        if force or self.KFtree.new_parent_flag:
            # 1.1-1.4 check close loop candidates
            if query_keyframe is None:
                close_loop_flag, close_loop_candidate_name, H_candidate_newest = self.check_closeloop_candidate()
            else:
                close_loop_flag, close_loop_candidate_name, H_candidate_newest = self.check_closeloop_candidate(query_keyframe)
            _logger.debug('%s', f"close loop flag is {close_loop_flag}")
            # 1.5 if true, continue
            if close_loop_flag:
                if query_keyframe is not None:
                    query_keyframe.update_features_and_mappoints()
                    self.KFtree.add_parent(query_keyframe)
                    for point in query_keyframe.get_mvpMapPoints().values():
                        point.add_observation(query_keyframe.mnId)
                _logger.debug('%s', f"close loop candidate is {close_loop_candidate_name}")
                # 2. correct loop by essential graph optimization and global BA
                # close loop target is the newest parent, reference frame
                optimized_pose, optimized_map = self.correct_loop(close_loop_candidate_name, H_candidate_newest)
                self.last_close_loop_keyframe_id = self.KFtree.parents[self.KFtree.newest_parent_name]["node"].kf.mnFrameId
                
                self.KFtree.grand_parents_kdtree.update_all_points(optimized_pose)

                return optimized_pose, optimized_map, close_loop_candidate_name
            
            else:
                return None, None, None


        return None, None, None


    def check_closeloop_candidate(self, query_keyframe=None):
        # 1.1 get the newest parent frame
        newest_parent_kf = query_keyframe
        if newest_parent_kf is None:
            newest_parent_kf = self.KFtree.parents[self.KFtree.newest_parent_name]["node"].get_kf()
        if self.last_loop_query_frame_id is not None and newest_parent_kf.mnFrameId <= self.last_loop_query_frame_id:
            return False, None, None
        self.last_loop_query_frame_id = newest_parent_kf.mnFrameId
        last_close_loop_interval = newest_parent_kf.mnFrameId - self.last_close_loop_keyframe_id
        _logger.debug('%s', f"frame interval is {last_close_loop_interval} for current kf {newest_parent_kf.mnFrameId} - last close loop {self.last_close_loop_keyframe_id}")
        if last_close_loop_interval < self.close_loop_cooldown_frames:
            self.pending_loop_candidate_frame_id = None
            self.pending_loop_candidate_count = 0
            _logger.debug('%s', f"loop query skipped: cooldown, remaining={self.close_loop_cooldown_frames - last_close_loop_interval}")
            return False, None, None
        newest_parent_kf_translation = newest_parent_kf.get_translation()
        newest_parent_xfeat = newest_parent_kf.get_xfeat_features()
        # Combine local pose neighbors with global appearance candidates. The
        # latter remains useful when drift puts a revisited area far away in
        # the estimated trajectory.
        spatial_names, _, _ = self.KFtree.grand_parents_kdtree.query_k_nearest_exclude_nearby_ids(
            newest_parent_kf_translation,
            newest_parent_kf.mnId,
            k=self.settings.spatial_candidates,
            id_margin=0,
            extra=self.settings.spatial_extra_candidates,
        )
        grandparent_keyframes = {
            name: node.get_kf()
            for name, node in self.KFtree.grand_parents.items()
        }
        appearance_names = [
            name for name, _ in self.rank_place_candidates(
                newest_parent_kf.image,
                grandparent_keyframes,
                newest_parent_kf.mnFrameId,
            )
        ]
        gp_names = list(dict.fromkeys(spatial_names + appearance_names))
        if not gp_names:
            self.pending_loop_candidate_frame_id = None
            self.pending_loop_candidate_count = 0
            return False, None, None
        close_loop_flag = False
        close_loop_candidate = None
        best_H_candidate_newest = None
        
        max_iou_closeloop_candidate = 0.0
        # 1.3 check if the parent kf has overlap with the grandparents
        for gp_name in gp_names:
            gp_kf = self.KFtree.grand_parents[gp_name].get_kf()
            gp_xfeat = gp_kf.get_xfeat_features()
            if gp_xfeat != None:
                xfeat_flag, H_candidate_newest = self.xfeat.match(gp_xfeat, newest_parent_xfeat)
                _logger.debug('%s', f"xfeat_flag is {xfeat_flag} for grandparent {gp_name} - {self.KFtree.newest_parent_name}")
                
                if xfeat_flag:
                    iou_close_loop = self.vis_con.compute_iou_from_homography_fast(H_candidate_newest)
                    
                    # two conditions for close loop
                    close_loop_condition_1 = iou_close_loop > self.close_loop_iou_threshold
                    _logger.debug('%s', f"iou_close_loop is {iou_close_loop} for grandparent {gp_name} - {self.KFtree.newest_parent_name}")
                    frame_interval = newest_parent_kf.mnFrameId - gp_kf.mnFrameId
                    close_loop_condition_2 = frame_interval >= self.close_loop_frame_interval
                    _logger.debug('%s', f"frame interval is {frame_interval} for grandparent {gp_name} - {self.KFtree.newest_parent_name}")
                    
                    # 1.4 if yes, set close loop flag true
                    if close_loop_condition_1 and close_loop_condition_2:
                        close_loop_flag = True
                        if iou_close_loop > max_iou_closeloop_candidate:
                            close_loop_candidate = gp_name
                            best_H_candidate_newest = H_candidate_newest
                            max_iou_closeloop_candidate = iou_close_loop

        if not close_loop_flag:
            self.pending_loop_candidate_frame_id = None
            self.pending_loop_candidate_count = 0
            return False, None, None

        candidate_frame_id = self.KFtree.grand_parents[close_loop_candidate].get_kf().mnFrameId
        if (
            self.pending_loop_candidate_frame_id is not None
            and abs(candidate_frame_id - self.pending_loop_candidate_frame_id) <= self.settings.confirmation_frame_tolerance
        ):
            self.pending_loop_candidate_count += 1
        else:
            self.pending_loop_candidate_frame_id = candidate_frame_id
            self.pending_loop_candidate_count = 1

        if self.pending_loop_candidate_count < self.loop_confirmation_count:
            _logger.debug('%s', "loop candidate pending confirmation "
                f"{self.pending_loop_candidate_count}/{self.loop_confirmation_count}")
            return False, None, None

        self.pending_loop_candidate_frame_id = None
        self.pending_loop_candidate_count = 0
        return True, close_loop_candidate, best_H_candidate_newest

    def find_relocalization_candidate(self, image, current_frame_id):
        all_keyframes = self.KFtree.get_all_keyframes_for_loopclosing()
        ranked = self.rank_place_candidates(
            image,
            all_keyframes,
            current_frame_id,
            limit=self.place_candidate_count,
        )
        if not ranked:
            return None, None, None

        current_xfeat = self.xfeat.detect(image)
        best_keyframe = None
        best_homography = None
        best_iou = 0.0
        for keyframe_id, appearance_score in ranked:
            candidate = all_keyframes[keyframe_id]
            candidate_xfeat = candidate.get_xfeat_features()
            if candidate_xfeat is None:
                continue
            matched, homography = self.xfeat.match(candidate_xfeat, current_xfeat)
            if not matched:
                continue
            iou = self.vis_con.compute_iou_from_homography_fast(homography)
            if iou > best_iou:
                best_keyframe = candidate
                best_homography = homography
                best_iou = iou
                _logger.debug('%s', f"relocalization candidate {keyframe_id}: "
                    f"appearance={appearance_score:.3f}, iou={iou:.3f}")

        if best_iou < self.relocalization_iou_threshold:
            return None, None, None
        return best_keyframe, best_homography, best_iou


    def correct_loop(self, close_loop_candidate_name, H_candidate_newest):
        # 2. prepare for loop closure
        target_id = self.KFtree.newest_parent_name
        ref_id = close_loop_candidate_name
        newest_parent_kf = self.KFtree.parents[target_id]["node"].get_kf()
        close_loop_candidate_kf = self.KFtree.grand_parents[close_loop_candidate_name].get_kf()
        # 2.1 clean the repeat features, and link old features to new frames
        T_newest_w_new, T_newest_candidate = self.compute_sim3(close_loop_candidate_kf, H_candidate_newest)
        # 2.2 load all grandparents kf and parents node for optimization
        all_keyframes = self.KFtree.get_all_keyframes_for_loopclosing()
        repaired_links = self.reconcile_map_observations(all_keyframes)
        if repaired_links:
            _logger.debug('%s', f"repaired {repaired_links} stale map observations before loop correction")
        _logger.debug('%s %s', "keyframes for closeloop: ", all_keyframes.keys())

        # 2.3 get relative pose between keyframes
        pose_from_original, relative_pairs_from_original = self.get_relative_pose(all_keyframes)

        # 2.4 prepare the data for pose graph optimization
        # create close loop pairs for update the relative pose
        close_loop_candidates = {}
        close_loop_candidate_pairs_relative = relative_pairs_from_original.copy()
        Rci = T_newest_candidate[:3,:3]
        tci = T_newest_candidate[:3, 3]
        relative_pose_key = f"{ref_id}-{target_id}"
        close_loop_candidate_pairs_relative[relative_pose_key] = {"Rci":Rci, "tci":tci, "ref":ref_id, "cur":target_id}

        close_loop_candidates[ref_id] = True
        close_loop_candidates[target_id] = True

        close_loop_candidate_pose = pose_from_original.copy()
        close_loop_candidate_pose[target_id] = T_newest_w_new

        # 2.5 calculate the features from the close loop candidate that can be seen in the ref frame
        # detele these features
        # link the close loop candidate with the ref frame features
        target_kf_new_feat_ids, redundant_feat_ids = self.update_features_before_correction(ref_id, target_id, all_keyframes)
        _logger.debug('%s %s', "redundant_feat_ids length is: ", len(redundant_feat_ids))
        _logger.debug('%s %s', "updated_features_for_target_frame length is: ", len(target_kf_new_feat_ids))

        # 2.7 check covisible frames with the feature id
        map_points_with_2d_feature_dict = self.load_map_points_with_2d_feature_dict(all_keyframes)
        co_visible_frames_dict = self.check_covisible_frames(map_points_with_2d_feature_dict, redundant_feat_ids)
        _logger.debug('%s', f"co_visible_frames_dict has {len(co_visible_frames_dict)} keyframes")
        
        # 2.8 update the feature from ref frame to the candidate frames
        self.update_close_loop_features_map_points(all_keyframes, ref_id, target_id, co_visible_frames_dict, redundant_feat_ids)


        # 2.6 calcuate the covisible frames that can do BA with ref frame
        close_loop_candidates_relative_pose_new, close_loop_candidates_w_pose_new = self.update_close_loop_candidates(all_keyframes, ref_id, target_id, co_visible_frames_dict)

        # 3.1 update the close loop candidate relative pose and world pose
        for key, new_relative_pose_value in close_loop_candidates_relative_pose_new.items():
            close_loop_candidate_pairs_relative[key] = new_relative_pose_value
            cur_id = new_relative_pose_value["cur"]
            close_loop_candidates[cur_id] = True

        for key, new_world_pose in close_loop_candidates_w_pose_new.items():
            close_loop_candidate_pose[key] = new_world_pose


        # # 3.2 use pose graph optimization to correct the close loop
        new_poses_after_optimization = self.optimizer.optimize_essential_graph_3(close_loop_candidate_pose, relative_pairs_from_original, close_loop_candidates)


        self.remove_redundant_map_points(all_keyframes, redundant_feat_ids)


        # update the pose and features after optimization
        updated_keyframe = self.update_pose_mappoints_after_loop_correct(all_keyframes, new_poses_after_optimization)

        map_points_with_2d_feature_dict_for_global_BA, corrected_cam_pose, pose_list = self.prepare_pose_mappoints_for_global_BA(updated_keyframe)

        optimized_pose, optimized_map = self.optimizer.local_bundle_adjustment_dense_2(corrected_cam_pose, map_points_with_2d_feature_dict_for_global_BA, ref_id, target_id)

        for key in all_keyframes.keys():
            if key in optimized_pose:
                all_keyframes[key].set_pose(optimized_pose[key])

        all_map_points = self.map.get_all_map_points()
        for key in optimized_map.keys():
            if key in all_map_points:
                all_map_points[key].set_world_pos(optimized_map[key].reshape(3,1))

        return optimized_pose, optimized_map


        '''

        keyframes = self.KFtree.load_keyframes()
        keyframes = self.load_keyframes_data(self.keyframe_save_file)
        # keyframes_ids = list(keyframes.keys())
        # keyframes[keyframes_ids[0]]["kf_pose"] = np.identity(4)

        map_points_with_2d_feature_dict = self.load_mappoint_data(self.map_points_save_file)

        cam_traj_dict = self.load_cam_traj_data(self.cam_traj_save_file)

        # 1.2 get relative pose between keyframes
        pose_from_original, relative_pairs_from_original = self.get_relative_pose(keyframes)
        pose_from_original_copy = pose_from_original.copy()

        # 1.3 find close loop
        ref_id, target_id, clp_H_target_ref = self.find_close_loop_offline(keyframes)

        # 1.4 get close loop ref frame and target frame
        clp_ref_frame = keyframes[ref_id]

        T_tar_w_new, T_tar_ref = self.compute_sim3(clp_ref_frame, clp_H_target_ref)

        # 2.1 prepare the data for pose graph optimization
        # create close loop pairs for update the relative pose
        close_loop_candidates = {}
        close_loop_candidate_pairs_relative = relative_pairs_from_original.copy()
        Rci = T_tar_ref[:3,:3]
        tci = T_tar_ref[:3, 3]
        relative_pose_key = f"{ref_id}-{target_id}"
        close_loop_candidate_pairs_relative[relative_pose_key] = {"Rci":Rci, "tci":tci, "ref":ref_id, "cur":target_id}

        close_loop_candidates[ref_id] = True
        close_loop_candidates[target_id] = True

        close_loop_candidate_pose = pose_from_original.copy()
        close_loop_candidate_pose[target_id] = T_tar_w_new

        # 2.2 calculate the features from the close loop candidate that can be seen in the ref frame
        # detele these features
        # link the close loop candidate with the ref frame features
        updated_features_for_target_frame, redundant_feat_ids = self.update_features_before_correction(ref_id, target_id, keyframes)
        print("redundant_feat_ids length is: ", len(redundant_feat_ids))
        print("updated_features_for_target_frame length is: ", len(updated_features_for_target_frame))
        # 2.3 update features for target frame
        keyframes[target_id]["feature"] = updated_features_for_target_frame
        # 2.4 check covisible frames with the feature id
        co_visible_frames_dict = self.check_covisible_frames(map_points_with_2d_feature_dict, redundant_feat_ids)
        print(f"co_visible_frames_dict is {len(co_visible_frames_dict)}")
        
        # 2.5 update the feature from ref frame to the candidate frames
        keyframes = self.update_close_loop_features_map_points(keyframes, ref_id, target_id, co_visible_frames_dict, redundant_feat_ids)


        # 3.0 get the interval id for more poses to have a better optimization result
        keyframe_ids_list = list(keyframes.keys())
        keyframe_length = len(keyframe_ids_list)
        interpolate_frame_ids = {}
        # get intermedian frame ids
        for idx in range(keyframe_length-2):
            cur_keyframe_id = keyframe_ids_list[idx]
            next_kf_id = keyframe_ids_list[idx+1]

            intermedian_id = int((next_kf_id - cur_keyframe_id)/2 + cur_keyframe_id)
            interpolate_frame_ids[intermedian_id] = {"cur":cur_keyframe_id, "next":next_kf_id}

        # num_samples = 20
        # for idx in range(keyframe_length - 2):
        #     cur_keyframe_id = keyframe_ids_list[idx]
        #     next_kf_id = keyframe_ids_list[idx + 1]

        #     # Generate multiple intermediate keyframe IDs
        #     interpolated_ids = np.linspace(cur_keyframe_id, next_kf_id, num_samples + 2, dtype=int)[1:-1]

        #     for intermedian_id in interpolated_ids:
        #         print(f"Interpolated Keyframe ID: {intermedian_id}")  # Replace with your processing logic
        #         interpolate_frame_ids[intermedian_id] = {"cur":cur_keyframe_id, "next":next_kf_id}
        # get the image and poses with the interpolate frames
        interpolate_image_dict = {}
        for frame_id in interpolate_frame_ids.keys():
            
            if frame_id in cam_traj_dict:
                interpolate_image = cam_traj_dict[frame_id]["image"]
                interpolate_cam_pose = cam_traj_dict[frame_id]["pose"]
                # use xfeat to detect the H matrix and propagate the features at here
                interpolate_cam_ref_1 = interpolate_frame_ids[frame_id]["cur"]
                interpolate_cam_ref_2 = interpolate_frame_ids[frame_id]["next"]

                ref_kf_ids = [interpolate_cam_ref_1, interpolate_cam_ref_2]
                img_interpolate_feat_dict = {}

                for ref_kf_id in ref_kf_ids:
                    kf_img = keyframes[ref_kf_id]["img"].copy()
                    kf_feature = keyframes[ref_kf_id]["feature"].copy()

                    kf_feature_list = list(kf_feature.values())
                    kf_feature_id_list = list(kf_feature.keys())

                    # calculate the H between reference kf and current frame
                    xfeat_flag, H, _ = self.xfeat.detect_and_match_2(kf_img, interpolate_image)
                    kf_features_interpolate = self.xfeat.warp_shitomasi_features(kf_feature_list, H).reshape(-1,2).tolist()

                    # check every features if in the image, if ture, keep the feature and id
                    for feat_key,feat_2d in zip(kf_feature_id_list, kf_features_interpolate):
                        x = feat_2d[0]
                        y = feat_2d[1]
                        if 0<x<720 and 0<y<720:
                            
                            img_interpolate_feat_dict[feat_key] = feat_2d
                            map_points_with_2d_feature_dict[feat_key]["feats"].append({frame_id:feat_2d.copy()})

                    # calculate relative pose
                    Sjw = np.linalg.inv(keyframes[ref_kf_id]["kf_pose"].copy())
                    Sji = Sjw @ interpolate_cam_pose
                    Rcw = Sji[:3,:3].copy()
                    tcw = Sji[:3, 3].copy()
                    relative_original_key = f"{ref_kf_id}-{frame_id}"

                    relative_pairs_from_original[relative_original_key] = {"Rci":Rcw, "tci":tcw, "ref":ref_kf_id, "cur":frame_id}
                
                interpolate_image_dict[frame_id] = {"kf_pose":interpolate_cam_pose, "img":interpolate_image, "feature":img_interpolate_feat_dict}


        # 2.6 calcuate the covisible frames that can do BA with ref frame
        close_loop_candidates_relative_pose_new, close_loop_candidates_w_pose_new = self.update_close_loop_candidates(keyframes, ref_id, target_id, co_visible_frames_dict)

        
        # 3.1 update the close loop candidate relative pose and world pose
        for key, new_relative_pose_value in close_loop_candidates_relative_pose_new.items():
            close_loop_candidate_pairs_relative[key] = new_relative_pose_value
            cur_id = new_relative_pose_value["cur"]
            close_loop_candidates[cur_id] = True

        for key, new_world_pose in close_loop_candidates_w_pose_new.items():
            close_loop_candidate_pose[key] = new_world_pose

        for frame_key in interpolate_image_dict.keys():
            close_loop_candidate_pose[frame_key] = interpolate_image_dict[frame_key]["kf_pose"]
            keyframes[frame_key] = interpolate_image_dict[frame_key]

        # # 3.2 use pose graph optimization to correct the close loop
        new_poses_after_optimization = optimize_essential_graph_3(close_loop_candidate_pose, relative_pairs_from_original, close_loop_candidates)

        # uncorrected_map = self.update_map_points_after_loop_correct(keyframes, new_poses_after_optimization)


        uncorrected_map_dict = {}
        for map_point_key, map_point_value in map_points_with_2d_feature_dict.items():
                uncorrected_map_dict[map_point_key] = map_point_value["map_point"]
        # for keyframe_value in keyframes.values():
        #     for map_point_key, map_point_value in keyframe_value["map_point"].items():
        #         uncorrected_map_dict[map_point_key] = map_point_value

        # for map_point_key, map_value in map_points_with_2d_feature_dict.items():
        #     uncorrected_map_dict[map_point_key] = map_value["map_point"]

        # correct_pose = []
        # for new_pose in new_poses_after_optimization.values():
        #     correct_pose.append(np.linalg.inv(new_pose)[:3,3])

        # cam_pos = []
        # for original_pose in pose_from_original_copy.values():
        #     cam_pos.append(np.linalg.inv(original_pose)[:3,3])

        # map_points_list = list(uncorrected_map_dict.values())
        # kf_map_points_all_old_arr = np.asarray(map_points_list)

        # self.plot_validation(cam_pos,correct_pose, kf_map_points_all_old_arr)


        # update the pose and features after optimization
        updated_keyframe = self.update_pose_mappoints_after_loop_correct(keyframes, new_poses_after_optimization, uncorrected_map_dict)


        map_points_with_2d_feature_dict_for_global_BA, corrected_cam_pose, pose_list = self.prepare_pose_mappoints_for_global_BA(updated_keyframe)

        optimized_pose, optimized_map = local_bundle_adjustment_dense_2(corrected_cam_pose, map_points_with_2d_feature_dict_for_global_BA, ref_id, target_id)


        t2 = time()

        print(f"the optimization used {t2 - t1} seconds.")

        optimized_map_list = list(optimized_map.values())
        optimized_map_arr = np.asarray(optimized_map_list)
        valid_indices = np.where(optimized_map_arr[:, 2] > 0.01)[0]

        valid_optimized_map_arr = optimized_map_arr[valid_indices]

        correct_pose = []
        for new_pose in optimized_pose.values():
            correct_pose.append(np.linalg.inv(new_pose)[:3,3])

        cam_pos = []
        for original_pose in pose_from_original_copy.values():
            cam_pos.append(np.linalg.inv(original_pose)[:3,3])

        map_points_list = list(uncorrected_map_dict.values())
        kf_map_points_all_old_arr = np.asarray(map_points_list)

        optimized_cam_traj_save_file = f"{self.data_folder}/optimized_cam_traj.pkl"
        with open(optimized_cam_traj_save_file, 'wb') as file:
            pickle.dump(optimized_pose, file)

        cam_traj_traj_image = {}
        for idx, key in enumerate(optimized_pose.keys()):
            print(f"key is {key}")
            cam_traj_image = cam_traj_dict[key]["image"].copy()
            cam_traj_pose = optimized_pose[key].copy()
            cam_traj_traj_image[key] = {"image":cam_traj_image, "pose":cam_traj_pose}

        cam_traj_traj_image_save_file = f"{self.data_folder}/optimized_cam_traj_image.pkl"
        with open(cam_traj_traj_image_save_file, 'wb') as file:
            pickle.dump(cam_traj_traj_image, file)
        
        # optimized_map_save_file = f"{self.data_folder}/optimized_map.pkl"
        # with open(optimized_map_save_file, 'wb') as file:
        #     pickle.dump(optimized_map, file)

        optimized_map_save_file = f"{self.data_folder}/optimized_map.npy"
        np.save(optimized_map_save_file, valid_optimized_map_arr)


        self.plot_validation(cam_pos,correct_pose, kf_map_points_all_old_arr, valid_optimized_map_arr)
    '''

    def compute_sim3(self, ref_frame, H_target_ref):
        # get the ref frame pose, 2d features, 3d map points
        # get the ref_frame pose
        Tc_ref_w = ref_frame.get_pose()
        # get the ref frame features
        kf_features_dict = ref_frame.get_KeysUn_dict()
        kf_features = list(kf_features_dict.values())
        # get the ref frame map points
        kf_map = ref_frame.get_mvpMapPoints()
        kf_map_points = []
        kf_map_points_ids = []
        for kf_mp_id, kf_mp in kf_map.items():
            kf_mp_pos = kf_mp.get_world_pos().reshape(-1)
            kf_map_points.append(kf_mp_pos)
            kf_map_points_ids.append(kf_mp_id)

        # transform the 3d map points in ref frame
        kf_map_points_arr = np.asarray(kf_map_points).reshape(-1,3)
        points_h = np.hstack((kf_map_points_arr, np.ones((kf_map_points_arr.shape[0], 1))))
        points_wrt_ref = (Tc_ref_w @ points_h.T).T
        map_in_cam_frame_dict = {}

        # transform the 2d features to target frame
        kf_features_in_target = self.xfeat.warp_shitomasi_features(kf_features, H_target_ref).reshape(-1,2).tolist()
        kf_features_in_target_dict = {}

        # create dictionaries of map points and features with id for the pose optimization in BA
        for key, map_point, feature in zip(kf_map_points_ids, points_wrt_ref, kf_features_in_target):
            kf_features_in_target_dict[key] = feature
            map_in_cam_frame_dict[key] = map_point.reshape(-1)[:3]

        # use BA calculate the target frame pose with respect to the ref frame
        n_initial_correspondences, R_tar_ref, t_tar_ref = self.optimizer.pose_optimization4(map_in_cam_frame_dict, kf_features_in_target_dict)
        T_tar_ref = np.identity(4)
        T_tar_ref[:3,:3] = R_tar_ref
        T_tar_ref[:3,3] = t_tar_ref

        T_tar_w_new = np.dot(T_tar_ref, Tc_ref_w)
        # return the relative pose between ref_frame and the target frame

        return T_tar_w_new, T_tar_ref

    def get_relative_pose(self, keyframes):
        around_image_id_list = [-2,-1,1,2]
        relative_pairs_from_original = {}
        pose_from_original = {}

        keyframes_ids = list(keyframes.keys())
        keyframe_length = len(keyframes)
        # set normal edges
        for idx, kf_ids in enumerate(keyframes_ids):
            nIDi = kf_ids
            Siw = keyframes[kf_ids].get_pose()
            pose_from_original[kf_ids] = Siw
            Swi = np.linalg.inv(Siw)

            for pKFj_id in around_image_id_list:
                jdx = idx + pKFj_id

                if jdx >= keyframe_length:
                    jdx -= keyframe_length

                nIDj = keyframes_ids[jdx]

                Sjw = keyframes[nIDj].get_pose()
                Sji = Sjw @ Swi
                Rcw = Sji[:3,:3]
                tcw = Sji[:3, 3]
                relative_original_key = f"{nIDi}-{nIDj}"
                # to be continue
                relative_pairs_from_original[relative_original_key] = {"Rci":Rcw, "tci":tcw, "ref":nIDi, "cur":nIDj}
        return pose_from_original, relative_pairs_from_original

    def update_features_before_correction(self, ref_id, target_id, all_keyframes):
        # update the observation for the target keyframe
        # get reference keyframe information
        ref_kf = all_keyframes[ref_id]
        ref_kf_xfeat = ref_kf.get_xfeat_features()
        ref_kf_feature_dict = ref_kf.get_KeysUn_dict()

        ref_kf_feature_list = list(ref_kf_feature_dict.values())
        ref_kf_feat_id_list = list(ref_kf_feature_dict.keys())

        # get target keyframe information
        target_kf = all_keyframes[target_id]
        target_kf_xfeat = target_kf.get_xfeat_features()
        target_kf_feature_dict = target_kf.get_KeysUn_dict()

        target_kf_feature_list = list(target_kf_feature_dict.values())
        target_kf_feat_id_list = list(target_kf_feature_dict.keys())

        xfeat_flag, H_last_first = self.xfeat.match(ref_kf_xfeat, target_kf_xfeat)
        if not xfeat_flag or H_last_first is None:
            return {}, {}
        kf_first_features_in_last = self.xfeat.warp_shitomasi_features(ref_kf_feature_list, H_last_first).reshape(-1,2).tolist()

        H_first_last = np.linalg.inv(H_last_first)
        kf_last_features_in_first = self.xfeat.warp_shitomasi_features(target_kf_feature_list, H_first_last).reshape(-1,2).tolist()

        last_kf_new_feat_ids = {}
        # add the feature from the first frame to the last frame
        for key, feats in zip(ref_kf_feat_id_list, kf_first_features_in_last):
            # check if the feature is in the image, if yes, add the feature to the last feature dict
            x = feats[0]
            y = feats[1]

            if 0 <= x < self.width and 0 <= y < self.height:
                # add new feature to the target frame
                update_map_point_from_ref = ref_kf.get_map_point(key)
                if target_kf.update_map_point(update_map_point_from_ref, feats, key):
                    last_kf_new_feat_ids[key] = feats
                    update_map_point_from_ref.add_observation(target_id)

        redundant_feat_ids = {}
        # delete the features from last frame that can be seen in first frame
        for key, feats in zip(target_kf_feat_id_list, kf_last_features_in_first):
            if key in ref_kf_feature_dict:
                continue
            x = feats[0]
            y = feats[1]
            if  x > 0 and x < self.width and y > 0 and y < self.height:
                
                # # delete feature from the target frame
                if self.map.get_map_point(key) is not None:
                    redundant_feat_ids[key] = feats
                

        return last_kf_new_feat_ids, redundant_feat_ids

    def load_map_points_with_2d_feature_dict(self, all_key_frames):
        all_map_points = self.map.get_all_map_points()
        map_points_with_2d_feature_dict = {}
        for map_key, map_value in all_map_points.items():
            map_point_pos = map_value.get_world_pos()
            map_point_obs = map_value.get_observations()
            for obs_frame_id in map_point_obs.keys():
                if obs_frame_id in all_key_frames:
                    observed_kf = all_key_frames[obs_frame_id]
                    observed_feat = observed_kf.get_KeysUn_dict_feature(map_key)
                    if observed_feat is None:
                        map_value.erase_observation(obs_frame_id)
                        continue

                    if map_key in map_points_with_2d_feature_dict:
                        map_points_with_2d_feature_dict[map_key]["map_point"] = map_point_pos.copy()
                        map_points_with_2d_feature_dict[map_key]["feats"].append({obs_frame_id:observed_feat.copy()})

                    else:
                        map_points_with_2d_feature_dict[map_key] = {"map_point":map_point_pos.copy(), "feats":[{obs_frame_id:observed_feat.copy()}]}
                
        return map_points_with_2d_feature_dict


    def check_covisible_frames(self, map_points_with_2d_feature_dict:dict, redundant_feat_ids):
        co_visible_frames_dict = {}
        for redundant_id in redundant_feat_ids:

            point_data = map_points_with_2d_feature_dict.get(redundant_id)
            if point_data is None:
                continue
            feat_with_frame_ids_dict = point_data["feats"]
        
            for feat_with_frame_id in feat_with_frame_ids_dict:
                frame_id, = feat_with_frame_id.keys()
                co_visible_frames_dict[frame_id] = True

        return co_visible_frames_dict

    def update_close_loop_features_map_points(self, keyframes, ref_id, target_id, covisible_frames_dict, redundant_feat_ids):
        covisible_frames_copy = covisible_frames_dict.copy()

        if ref_id in covisible_frames_copy:
            del covisible_frames_copy[ref_id]

        covisible_frames_ids = list(covisible_frames_copy.keys())
        sorted_frame_ids = sorted(covisible_frames_ids)
        
        reversed_keyframes_ids = sorted_frame_ids[::-1]
        if not reversed_keyframes_ids:
            return
        last_keyframes_id = reversed_keyframes_ids[0]


        covisible_frames_size = len(covisible_frames_ids)
        # start backpropagate the features from ref frame
        if last_keyframes_id == target_id:
            for idx, frame_id in enumerate(reversed_keyframes_ids):
                jdx = idx + 1
                if jdx < covisible_frames_size:
                    idx_kf = keyframes[frame_id]
                    idx_kf_xfeat = idx_kf.get_xfeat_features()
                    idx_kf_feature = idx_kf.get_KeysUn_dict()
                    idx_kf_feature_list = list(idx_kf_feature.values())
                    idx_kf_feat_id_list = list(idx_kf_feature.keys())
                    
                    frame_id_jdx = reversed_keyframes_ids[jdx]

                    jdx_kf = keyframes[frame_id_jdx]
                    jdx_kf_xfeat = jdx_kf.get_xfeat_features()
                    jdx_kf_feature = jdx_kf.get_KeysUn_dict()


                    xfeat_flag, H_target_ref = self.xfeat.match(idx_kf_xfeat, jdx_kf_xfeat)
                    if not xfeat_flag or H_target_ref is None:
                        continue
                    kf_idx_features_in_jdx = self.xfeat.warp_shitomasi_features(idx_kf_feature_list, H_target_ref).reshape(-1,2).tolist()


                    # add the feature from the target frame
                    for key, feats in zip(idx_kf_feat_id_list, kf_idx_features_in_jdx):
                        x = feats[0]
                        y = feats[1]
                        if  x >= 0 and x < self.width and y >= 0 and y <= self.height:

                            update_map_point_from_ref = self.map.get_map_point(key)
                            if jdx_kf.update_map_point(update_map_point_from_ref, feats, key):
                                update_map_point_from_ref.add_observation(frame_id_jdx)


        return


    def update_close_loop_candidates(self, keyframes, ref_id, target_id, covisible_frames_dict):

        if ref_id in covisible_frames_dict:
            del covisible_frames_dict[ref_id]

        if target_id in covisible_frames_dict:
            del covisible_frames_dict[target_id]

        keyframes_ids = list(covisible_frames_dict.keys())
        _logger.debug('%s %s', "update_close_loop_candidates: ", keyframes_ids)
        reverse_keyframes_ids = reversed(sorted(keyframes_ids))

        ref_kf = keyframes[ref_id]
        ref_xfeat = ref_kf.get_xfeat_features()

        Tc_ref_w = ref_kf.get_pose()

        ref_map_points_values = []
        ref_map_points_ids = list(ref_kf.get_mvpMapPoints().keys())
        _logger.debug('%s %s', "ref_map_points_ids ", ref_map_points_ids)

        for key in ref_map_points_ids:
            ref_map_point_obj = self.map.get_map_point(key)
            if ref_map_point_obj is None:
                continue
            ref_map_point = ref_map_point_obj.get_world_pos().reshape(-1)
            ref_map_points_values.append(ref_map_point)

        
        _logger.debug('%s %s', "length ref_map_points_values ", len(ref_map_points_values))
        # transform the 3d map points in ref frame
        kf_map_points_arr = np.asarray(ref_map_points_values)
        points_h = np.hstack((kf_map_points_arr, np.ones((kf_map_points_arr.shape[0], 1))))
        points_wrt_ref = (Tc_ref_w @ points_h.T).T

        map_in_cam_frame_dict = {}

        # create dictionaries of map points and features with id for the pose optimization in BA
        valid_ref_ids = [
            key for key in ref_map_points_ids
            if self.map.get_map_point(key) is not None
        ]
        for key, local_map_point in zip(valid_ref_ids, points_wrt_ref):
            
            map_in_cam_frame_dict[key] = local_map_point.reshape(-1)[:3]

        close_loop_candidates_relative_pose_new = {}
        close_loop_candidates_w_pose_new = {}

        for cur_id in reverse_keyframes_ids:
            _logger.debug('%s %s', "update_close_loop_candidates cur_id: ", cur_id)
            cur_kf = keyframes[cur_id]
            cur_xfeat = cur_kf.get_xfeat_features()
            

            xfeat_flag, H_target_ref = self.xfeat.match(ref_xfeat, cur_xfeat)
            if xfeat_flag:
                iou = self.vis_con.compute_iou_from_homography_fast(H_target_ref)

                if iou > self.covisible_iou_threshold:
                    # if covisible > 50%, then calculate relative pose
                    # get target frame features
                    features_in_cur_dict = cur_kf.get_KeysUn_dict()
                    _logger.debug('%s %s', "length of features_in_cur_dict: ", len(features_in_cur_dict))
                    _logger.debug('%s %s', "features_in_cur_dict ids ", features_in_cur_dict.keys())
                    # calculate relative pose with BA
                    n_initial_correspondences, R_tar_ref, t_tar_ref = self.optimizer.pose_optimization4(map_in_cam_frame_dict, features_in_cur_dict)
                    T_tar_ref = np.identity(4)
                    T_tar_ref[:3,:3] = R_tar_ref
                    T_tar_ref[:3,3] = t_tar_ref

                    T_tar_w_new = np.dot(T_tar_ref, Tc_ref_w)

                    relative_pose_key = f"{ref_id}-{cur_id}"
                    close_loop_candidates_relative_pose_new[relative_pose_key] = {"Rci":R_tar_ref, "tci":t_tar_ref, "ref":ref_id, "cur":cur_id}

                    close_loop_candidates_w_pose_new[cur_id] = T_tar_w_new
                else:
                    _logger.debug('%s %s', "covisible < 0.5, frame id is ", cur_id)
                    break

        return close_loop_candidates_relative_pose_new, close_loop_candidates_w_pose_new


    def update_pose_mappoints_after_loop_correct(self, keyframe, keyframe_after_loop_correct):

        for kf_id in keyframe.keys():

            # udpate the new pose after loop correction
            new_world_frame_pose = keyframe_after_loop_correct[kf_id]

            keyframe[kf_id].set_pose(new_world_frame_pose)

        return keyframe
    
    def prepare_pose_mappoints_for_global_BA(self, updated_keyframe):
        map_points_with_2d_feature_dict_for_global_BA = {}
        corrected_cam_pose = {}
        pose_list = []

        for updated_keyframe_key, values in updated_keyframe.items():
            kf_pose = values.get_pose()

            pose_list.append(np.linalg.inv(kf_pose)[:3,3])

            map_points = values.get_mvpMapPoints()

            kf_feature = values.get_KeysUn_dict()

            corrected_cam_pose[updated_keyframe_key] = kf_pose

            for map_key, map_value in map_points.items():
                if map_value is None or map_key not in kf_feature:
                    continue
                feature_in_frame = kf_feature[map_key].copy()
                map_point_pos = map_value.get_world_pos().reshape(-1)
                if map_key in map_points_with_2d_feature_dict_for_global_BA:
                    map_points_with_2d_feature_dict_for_global_BA[map_key]["map_point"] = map_point_pos
                    map_points_with_2d_feature_dict_for_global_BA[map_key]["feats"].append({updated_keyframe_key:feature_in_frame})
                else:
                    map_points_with_2d_feature_dict_for_global_BA[map_key] = {"map_point":map_point_pos, "feats":[{updated_keyframe_key:feature_in_frame}]}

        return map_points_with_2d_feature_dict_for_global_BA, corrected_cam_pose, pose_list

    def remove_redundant_map_points(self, all_keyframes, redundant_feat_ids):
        """Remove merged points from every known observer without assuming dense IDs."""
        for map_point_id in list(redundant_feat_ids):
            map_point = self.map.get_map_point(map_point_id)
            observed_frame_ids = (
                list(map_point.get_observations()) if map_point is not None else []
            )

            # Observation lists can already be stale after an earlier closure,
            # so scan the supplied keyframes as well as the recorded observers.
            frame_ids = set(observed_frame_ids)
            frame_ids.update(
                frame_id
                for frame_id, keyframe in all_keyframes.items()
                if map_point_id in keyframe.get_mvpMapPoints()
            )

            for frame_id in frame_ids:
                keyframe = all_keyframes.get(frame_id)
                if keyframe is not None:
                    keyframe.delete_map_point_match(map_point_id)
                if map_point is not None:
                    map_point.erase_observation(frame_id)

            if map_point is not None:
                map_point.clear_all_observations()
            self.map.erase_map_point(map_point_id)

    def reconcile_map_observations(self, all_keyframes):
        """Repair stale links while preserving observations outside this closure set."""
        repairs = 0
        for frame_id, keyframe in all_keyframes.items():
            frame_points = keyframe.get_mvpMapPoints()
            frame_features = keyframe.get_KeysUn_dict()
            for map_point_id, frame_point in frame_points.items():
                map_point = self.map.get_map_point(map_point_id)
                if map_point is None or map_point_id not in frame_features:
                    keyframe.delete_map_point_match(map_point_id)
                    if frame_point is not None:
                        frame_point.erase_observation(frame_id)
                    repairs += 1
                    continue

                if frame_point is not map_point:
                    keyframe.update_map_point(
                        map_point,
                        frame_features[map_point_id],
                        map_point_id,
                    )
                    repairs += 1
                if frame_id not in map_point.get_observations():
                    map_point.add_observation(frame_id)
                    repairs += 1

        for map_point_id, map_point in self.map.get_all_map_points().items():
            for frame_id in list(map_point.get_observations()):
                keyframe = all_keyframes.get(frame_id)
                if keyframe is None:
                    continue
                if map_point_id not in keyframe.get_mvpMapPoints():
                    map_point.erase_observation(frame_id)
                    repairs += 1
        return repairs


    #         # update the mappoints in the keyframe


