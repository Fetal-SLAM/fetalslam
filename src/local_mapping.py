import numpy as np
import cv2

from .map_point import MapPoint
from .visual_reconstruction import VisualReconstruction
from .features.detector import Xfeat_Det_Matching
from .keyframe import KeyFrameTree
from .optimizer import Optimizer
from .map import Mapping

import logging as _logging

_logger = _logging.getLogger(__name__)


from .runtime_config import runtime_settings


class LocalMapping:
    def __init__(self, width, height, max_fake_features=200, settings=None):
        self.settings = settings or runtime_settings("local_mapping")
        self.mlNewKeyFrame = None
        self.width = width
        self.height = height
        self.max_fake_features = max_fake_features

        self.new_reconstruction_threshold = self.settings.new_reconstruction_threshold

        self.covisible_repeat_threshold = self.settings.covisible_repeat_threshold

        self.new_map_points = []
        self.reused_feature_ids = set()
        self.local_point_reuse_enabled = False
        self.min_parallax_cos = float(np.cos(np.deg2rad(1.0)))
        self.max_depth_factor = 5.0
        self.distance_ratio_factor = 1.5

    def configure_triangulation(
        self, min_parallax_degrees, max_depth_factor, distance_ratio_factor
    ):
        self.min_parallax_cos = float(
            np.cos(np.deg2rad(float(min_parallax_degrees)))
        )
        self.max_depth_factor = float(max_depth_factor)
        self.distance_ratio_factor = float(distance_ratio_factor)

    def set_vis_recon(self, vis_con:VisualReconstruction):
        self.vis_con = vis_con #Class for visual reconstruction

    def set_KeyFrameTree(self, KFtree:KeyFrameTree):
        self.KFtree = KFtree

    def set_xfeat_detector(self, xfeat:Xfeat_Det_Matching):
        self.xfeat = xfeat

    def set_optimizer(self, optimizer:Optimizer):
        self.optimizer = optimizer

    def set_map(self, global_map:Mapping):
        self.map = global_map

    def run(self):
        new_frame_flag, latest_child_name, best_candidate_name, H_latest_child_best_candidate, children_H = self.check_new_key_frame()
        reused_observations = 0
        self.reused_feature_ids = set()
        if new_frame_flag:
            if self.local_point_reuse_enabled:
                parent = self.KFtree.parents[self.KFtree.newest_parent_name]
                previous_ids = set(parent[best_candidate_name].kf.get_KeysUn_dict())
                reused_observations = self.associate_existing_points(parent[best_candidate_name].kf)
                self.reused_feature_ids = set(parent[best_candidate_name].kf.get_KeysUn_dict()) - previous_ids
            # assign new features and reconstruct them
            new_reconstruction_flag, new_map_points = self.assign_new_features_and_reconstruct(latest_child_name, best_candidate_name, H_latest_child_best_candidate)
            if new_reconstruction_flag:
                # update the map points to other child
                self.update_map_points_to_other_child(new_map_points, best_candidate_name, children_H)
                # local bundle adjustment
                optimized_frame_poses, optimized_map_points = self.optimizer.local_bundle_adjustment(self.KFtree)
                self.KFtree.update_all_keyframe_pos(optimized_frame_poses)

                # filter the outlier map points
                # update optimized map points
                self.map.update_map_points_world_pos(optimized_map_points)


                # # culling map points


            # update the keyframe tree (parent node, grandparent node)
            self.KFtree.update_keyframe_tree(
                new_reconstruction_flag or reused_observations >= self.settings.reuse_min_observations, best_candidate_name
            )

            # update the parent node
            self.KFtree.update_node()

        return {
            "processed": new_frame_flag,
            "new_parent": bool(self.KFtree.new_parent_flag),
            "new_parent_name": self.KFtree.newest_parent_name,
            "reused_observations": reused_observations,
        }


    def associate_existing_points(self, target):
        """Attach measured local observations without changing map geometry or IDs."""
        tracked = getattr(target, "mvTrackedPoints", None)
        active_count = np.count_nonzero(tracked) if tracked is not None else len(target.get_KeysUn_dict())
        budget = min(self.settings.reuse_max_points, self.max_fake_features - active_count)
        if budget <= 0:
            return 0
        target_gray = target.image
        if target_gray.ndim == 3:
            target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
        height, width = target_gray.shape
        pose = target.get_pose()
        added = 0
        # Distant historical search remains the responsibility of LoopClosing.
        sources = list(self.KFtree.parents.values())[-self.settings.reuse_parent_count:]
        for parent in reversed(sources):
            source = parent["node"].get_kf()
            if source.mnId == target.mnId:
                continue
            existing = target.get_KeysUn_dict()
            candidates, source_pixels, predictions = [], [], []
            source_features = source.get_KeysUn_dict()
            for point_id, point in source.get_mvpMapPoints().items():
                if point_id in existing or point_id not in source_features or point.is_bad():
                    continue
                world = np.asarray(point.get_world_pos()).reshape(3)
                camera = pose[:3, :3] @ world + pose[:3, 3]
                if not np.all(np.isfinite(camera)) or camera[2] <= 0:
                    continue
                pixel = (self.optimizer.mK @ camera)[:2] / camera[2]
                source_pixel = np.asarray(source_features[point_id]).reshape(2)
                if not np.all(np.isfinite(source_pixel)):
                    continue
                if not (self.settings.reuse_pixel_margin <= pixel[0] < width - self.settings.reuse_pixel_margin and self.settings.reuse_pixel_margin <= pixel[1] < height - self.settings.reuse_pixel_margin):
                    continue
                candidates.append((point_id, point))
                source_pixels.append(source_pixel)
                predictions.append(pixel)
            if not candidates:
                continue
            source_gray = source.image
            if source_gray.ndim == 3:
                source_gray = cv2.cvtColor(source_gray, cv2.COLOR_BGR2GRAY)
            if source_gray.shape != target_gray.shape:
                continue
            previous = np.asarray(source_pixels, dtype=np.float32).reshape(-1, 1, 2)
            predicted = np.asarray(predictions, dtype=np.float32).reshape(-1, 1, 2)
            measured, forward_ok, error = cv2.calcOpticalFlowPyrLK(
                source_gray, target_gray, previous, predicted.copy(),
                winSize=(self.settings.flow_window, self.settings.flow_window), maxLevel=self.settings.flow_levels, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )
            if measured is None:
                continue
            backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
                target_gray, source_gray, measured, previous.copy(),
                winSize=(self.settings.flow_window, self.settings.flow_window), maxLevel=self.settings.flow_levels, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )
            if backward is None:
                continue
            valid = forward_ok.ravel().astype(bool) & backward_ok.ravel().astype(bool)
            valid &= np.linalg.norm((backward - previous).reshape(-1, 2), axis=1) <= self.settings.flow_back_error
            valid &= np.linalg.norm((measured - predicted).reshape(-1, 2), axis=1) <= self.settings.flow_projection_error
            valid &= error.ravel() <= self.settings.flow_photometric_error
            occupied = [np.asarray(p).reshape(2) for p in existing.values()]
            for index in np.flatnonzero(valid):
                pixel = measured[index, 0]
                if not np.all(np.isfinite(pixel)):
                    continue
                if not (self.settings.reuse_pixel_margin <= pixel[0] < width - self.settings.reuse_pixel_margin and self.settings.reuse_pixel_margin <= pixel[1] < height - self.settings.reuse_pixel_margin):
                    continue
                if occupied and np.min(np.linalg.norm(np.asarray(occupied) - pixel, axis=1)) < self.settings.reuse_min_spacing:
                    continue
                point_id, point = candidates[index]
                if target.update_map_point(point, pixel, point_id):
                    point.add_observation(target.mnId)
                    point.local_recovery_protected = True
                    occupied.append(pixel.copy())
                    added += 1
                if added >= budget:
                    break
            if added >= budget:
                break
        if added:
            _logger.debug('%s', f"local mapping reused {added} observations in KF {target.mnId}")
        return added

    def check_new_key_frame(self):
        """
        Check the covisible area of the children in the newest parent node.

        """
        # Get the newest parent (last key in OrderedDict)


        newest_parent = self.KFtree.newest_parent_name

        children_name = self.KFtree.parents[newest_parent]["children"]

        if len(children_name) < 2:
            return False, None, None, None, None

        # when the newest parent has more than two children
        # check if the newest child can be reconstructed
        best_candidate_name = None
        H_latest_child_best_candidate = None
        children_H = {}
        for child_name in children_name:
            H_parent_child = self.KFtree.parents[newest_parent][child_name].get_H_ref_cur()
            children_H[child_name] = H_parent_child
        # get the covisible area among children
        latest_child_name = children_name[-1]
        latest_parent_child_H = children_H[latest_child_name]

        # check if this area repeats compared with other children
        H_latest_child_parent = np.linalg.inv(latest_parent_child_H)
        max_iou = 0.0
        for name in children_H.keys():
            if name != latest_child_name:
                H_parent_child_temp = children_H[name]
                H_latest_child_temp = H_latest_child_parent @ H_parent_child_temp
                iou_temp = self.vis_con.compute_iou_from_homography_fast(H_latest_child_temp)

                if iou_temp > max_iou:
                    max_iou = iou_temp
                    best_candidate_name = name
                    H_latest_child_best_candidate = H_latest_child_temp

        return best_candidate_name is not None, latest_child_name, best_candidate_name, H_latest_child_best_candidate, children_H


    def assign_new_features_and_reconstruct(self, latest_child_name, best_candidate_name, H_latest_child_best_candidate):

        new_points_flag = False
        new_map_points = None
        filtered_features_flag = False
        newest_parent = self.KFtree.newest_parent_name


        latest_child_kf = self.KFtree.parents[newest_parent][latest_child_name].kf
        best_candidate_child_kf = self.KFtree.parents[newest_parent][best_candidate_name].kf

        new_features_counter = self.max_fake_features - np.count_nonzero(best_candidate_child_kf.mvTrackedPoints)

        H_parent_best_candidate = self.KFtree.parents[newest_parent][best_candidate_name].get_H_ref_cur()


        roi_parent_best_candidate = self.vis_con.warp_iou(H_parent_best_candidate)
        roi_latest_child_best_candidate = self.vis_con.warp_iou(H_latest_child_best_candidate)


        iou, intersection, non_overlap_coords_1, non_overlap_coords_2 = self.vis_con.compute_iou_from_rois(roi_parent_best_candidate, roi_latest_child_best_candidate)
        new_areas = non_overlap_coords_2

        if new_features_counter <= 0:
            return False, None

        new_reconstruct_area = self.create_mask(new_areas)
        for point_id in self.reused_feature_ids:
            feature = best_candidate_child_kf.get_KeysUn_dict_feature(point_id)
            x, y = np.asarray(feature).reshape(2)
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(new_reconstruct_area, (int(round(x)), int(round(y))), 4, 0, -1)

        new_features_flag, new_features = self.xfeat.detect_shitomasi_pyramid(best_candidate_child_kf.image, start_quality=self.settings.corner_quality, max_features=new_features_counter, min_quality=self.settings.corner_min_quality, mask=new_reconstruct_area)

        # check if the new features are valid
        # filter new map points in other parents
        if new_features_flag:
            filtered_features_flag, filtered_features = self.check_candidate_with_other_parents(new_features, best_candidate_name)

        if filtered_features_flag:


            best_candidate_child_kf.new_mvKeysUn = filtered_features.astype(int)
            H_tar_cur = np.linalg.inv(H_latest_child_best_candidate)
            latest_child_kf.new_mvKeysUn = self.warp_features(filtered_features, H_tar_cur)

            new_points_flag, new_map_points = self.create_new_map_points(best_candidate_child_kf, latest_child_kf)
            if new_points_flag:
                best_candidate_child_kf.update_after_triangulation(new_map_points)
                self.map.add_map_points(new_map_points)

                valid_KeysUn_dict = best_candidate_child_kf.get_KeysUn_dict().keys()

                frame_id = best_candidate_child_kf.mnId

                self.map.add_observations_in_map(frame_id,valid_KeysUn_dict)


        return new_points_flag, new_map_points


    def check_candidate_with_other_parents(self, new_features, best_candidate_name):
        # check if have high overlap with other parent nodes

        # load the best candidate node
        newest_parent = self.KFtree.newest_parent_name
        best_candidate_kf = self.KFtree.parents[newest_parent][best_candidate_name].get_kf()
        best_candidate_xfeat = best_candidate_kf.get_xfeat_features()

        for parent_name in self.KFtree.parents.keys():
            if parent_name == newest_parent:
                continue

            if len(new_features) <= 0:
                break

            other_parent_xfeat = self.KFtree.parents[parent_name]["node"].kf.get_xfeat_features()
            if other_parent_xfeat != None:
                xfeat_flag, H_other_parent_candidate = self.xfeat.match(other_parent_xfeat, best_candidate_xfeat)
                if xfeat_flag:
                    H_candidate_other_parents = np.linalg.inv(H_other_parent_candidate)

                    new_features_parent = self.warp_features(new_features, H_candidate_other_parents)
                    invalid_new_features = self.vis_con.check_features_visibility(new_features_parent.reshape(-1,2)).reshape(-1)

                    # filter the repeated features
                    new_features = new_features[~invalid_new_features]

        if len(new_features) > 0:
            return True, new_features
        else:
            return False, None

    def update_map_points_to_other_child(self, new_map_points, best_candidate_name, children_H):
        best_candidate_kf = self.KFtree.parents[self.KFtree.newest_parent_name][best_candidate_name].kf
        valid_new_features = best_candidate_kf.new_mvKeysUn

        H_parent_best_candidate = children_H[best_candidate_name]

        # use children_H to warp the new map points to other children
        for child_name in children_H.keys():
            if child_name != best_candidate_name:

                H_parent_child = children_H[child_name]
                H_best_candidate_other_child = np.linalg.inv(H_parent_best_candidate) @ H_parent_child
                new_features_child = self.warp_features(valid_new_features, H_best_candidate_other_child)
                self.KFtree.parents[self.KFtree.newest_parent_name][child_name].kf.new_mvKeysUn = new_features_child
                # add new map points to the child node
                self.KFtree.parents[self.KFtree.newest_parent_name][child_name].kf.update_after_triangulation(new_map_points)

                # update the observation to new child node
                valid_KeysUn_dict = self.KFtree.parents[self.KFtree.newest_parent_name][child_name].kf.get_KeysUn_dict().keys()

                frame_id = self.KFtree.parents[self.KFtree.newest_parent_name][child_name].kf.mnId

                self.map.add_observations_in_map(frame_id,valid_KeysUn_dict)


    def create_new_map_points(self,mpTargetKeyFrame, mpCurrentKeyFrame):
        # get the new key frame
        Rcw1 = mpTargetKeyFrame.get_rot()
        Rwc1 = Rcw1.T
        tcw1 = mpTargetKeyFrame.get_translation()
        Tcw1 = mpTargetKeyFrame.get_pose()

        Ow1 = mpTargetKeyFrame.get_camera_center()
        fx1 = mpTargetKeyFrame.fx
        fy1 = mpTargetKeyFrame.fy
        cx1 = mpTargetKeyFrame.cx
        cy1 = mpTargetKeyFrame.cy
        invfx1 = mpTargetKeyFrame.invfx
        invfy1 = mpTargetKeyFrame.invfy

        ratioFactor = self.distance_ratio_factor

        nnew = 0

        # get current frame
        Rcw2 = mpCurrentKeyFrame.get_rot()
        Rwc2 = Rcw2.T
        tcw2 = mpCurrentKeyFrame.get_translation()
        Tcw2 = mpCurrentKeyFrame.get_pose()
        Ow2 = -np.dot(Rwc2, tcw2)
        fx2 = mpCurrentKeyFrame.fx
        fy2 = mpCurrentKeyFrame.fy
        cx2 = mpCurrentKeyFrame.cx
        cy2 = mpCurrentKeyFrame.cy
        invfx2 = mpCurrentKeyFrame.invfx
        invfy2 = mpCurrentKeyFrame.invfy


        vBaseline = Ow2-Ow1
        baseline = np.linalg.norm(vBaseline)

        medianDepthKF1 = mpTargetKeyFrame.compute_scene_median_depth(2)

        medianDepthKF1 = float(np.asarray(medianDepthKF1).reshape(-1)[0])
        if not np.isfinite(medianDepthKF1) or medianDepthKF1 <= 0:
            _logger.warning('%s', "[localmapping] invalid median scene depth.")
            return False, None
        ratioBaselineDepth = baseline/medianDepthKF1

        if ratioBaselineDepth < self.settings.min_baseline_depth_ratio:
            _logger.debug('%s', "[localmapping] ratioBaselineDepth is too small for new points trianglation.")
            return False, None

        # get the new fake features

        vKeys1 = mpTargetKeyFrame.new_mvKeysUn

        vKeys2 = mpCurrentKeyFrame.new_mvKeysUn

        vMatchedIndices = [cv2.DMatch(i,i,0) for i in range(len(vKeys1))]

        mp_index_tar = mpTargetKeyFrame.N

        # Triangulate each match
        nmatches = len(vMatchedIndices)
        new_map_points = [None] * nmatches
        rejected = {
            "parallax": 0,
            "homogeneous": 0,
            "cheirality": 0,
            "depth": 0,
            "reprojection": 0,
            "distance": 0,
        }

        for ikp in range(nmatches):

            # Extract keypoints and their properties
            kp1 = vKeys1[ikp]
            kp1_ur = -1
            bStereo1 = kp1_ur >= 0

            kp2 = vKeys2[ikp]
            kp2_ur = -1
            bStereo2 = kp2_ur >= 0

            # Check parallax between rays
            xn1 = np.array([(kp1[0][0] - cx1) * invfx1,
                            (kp1[0][1] - cy1) * invfy1,
                            1.0]).reshape(3, 1)
            xn2 = np.array([(kp2[0][0] - cx2) * invfx2,
                            (kp2[0][1] - cy2) * invfy2,
                            1.0]).reshape(3, 1)

            ray1 = np.dot(Rwc1, xn1)
            ray2 = np.dot(Rwc2, xn2)
            cosParallaxRays = float(
                np.asarray(
                    np.dot(ray1.T, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))
                ).reshape(-1)[0]
            )
            cosParallaxRays = float(np.clip(cosParallaxRays, -1.0, 1.0))

            cosParallaxStereo = cosParallaxRays + 1
            cosParallaxStereo1 = cosParallaxStereo
            cosParallaxStereo2 = cosParallaxStereo

            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2)


            if 0 < cosParallaxRays < self.min_parallax_cos:
                # Linear triangulation
                A = np.zeros((4, 4))
                A[0] = xn1[0] * Tcw1[2, :] - Tcw1[0, :]
                A[1] = xn1[1] * Tcw1[2, :] - Tcw1[1, :]
                A[2] = xn2[0] * Tcw2[2, :] - Tcw2[0, :]
                A[3] = xn2[1] * Tcw2[2, :] - Tcw2[1, :]

                _, _, vt = np.linalg.svd(A)
                x3D = vt[-1].reshape(4, 1)

                if abs(x3D[3, 0]) <= 1e-9:
                    rejected["homogeneous"] += 1
                    continue

                x3D = x3D[:3] / x3D[3]
            else:
                rejected["parallax"] += 1
                continue

            x3Dt = x3D.T

            # Check if points are in front of the cameras
            z1 = np.dot(Rcw1[2], x3Dt.reshape(-1)) + tcw1[2]
            if z1 <= 0:
                rejected["cheirality"] += 1
                continue

            z2 = np.dot(Rcw2[2], x3Dt.reshape(-1)) + tcw2[2]
            if z2 <= 0:
                rejected["cheirality"] += 1
                continue
            max_depth = self.max_depth_factor * medianDepthKF1
            if float(z1) > max_depth or float(z2) > max_depth:
                rejected["depth"] += 1
                continue

            # Check reprojection errors
            sigmaSquare1 = 1.0
            x1 = np.dot(Rcw1[0], x3Dt.reshape(-1)) + tcw1[0]
            y1 = np.dot(Rcw1[1], x3Dt.reshape(-1)) + tcw1[1]
            invz1 = 1.0 / z1

            if not bStereo1:
                u1 = fx1 * x1 * invz1 + cx1
                v1 = fy1 * y1 * invz1 + cy1
                errX1 = u1 - kp1[0][0]
                errY1 = v1 - kp1[0][1]
                if (errX1**2 + errY1**2) > self.settings.reprojection_chi2 * sigmaSquare1:
                    rejected["reprojection"] += 1
                    continue


            sigmaSquare2 = 1.0
            x2 = np.dot(Rcw2[0], x3Dt.reshape(-1)) + tcw2[0]
            y2 = np.dot(Rcw2[1], x3Dt.reshape(-1)) + tcw2[1]
            invz2 = 1.0 / z2

            if not bStereo2:
                u2 = fx2 * x2 * invz2 + cx2
                v2 = fy2 * y2 * invz2 + cy2
                errX2 = u2 - kp2[0][0]
                errY2 = v2 - kp2[0][1]
                if (errX2**2 + errY2**2) > self.settings.reprojection_chi2 * sigmaSquare2:
                    rejected["reprojection"] += 1
                    continue
            normal1 = x3D - Ow1
            dist1 = np.linalg.norm(normal1)
            normal2 = x3D - Ow2
            dist2 = np.linalg.norm(normal2)

            if dist1 == 0 or dist2 == 0:
                rejected["distance"] += 1
                continue

            ratioDist = dist2/dist1
            # A distance-ratio gate requires feature octave/scale information.
            # The generated Shi-Tomasi points are single-scale, so 0 disables
            # this optional check while the geometric checks above remain.
            if ratioFactor > 1.0 and (
                ratioDist < 1.0 / ratioFactor or ratioDist > ratioFactor
            ):
                rejected["distance"] += 1
                continue


            pMP = MapPoint(x3D, mpTargetKeyFrame)
            pMP.set_map_point_color(mpTargetKeyFrame.get_feature_color((kp1[0][0], kp1[0][1])))
            new_map_points[ikp] = pMP

            pMP.add_observation(mpTargetKeyFrame.mnId)


            nnew += 1

        _logger.debug('%s', "[localmapping] "
            f"matches={nmatches}, new={nnew}, baseline={baseline:.6f}, "
            f"median_depth={medianDepthKF1:.6f}, ratio={ratioBaselineDepth:.6f}, "
            f"rejected={rejected}")
        if nnew > self.settings.min_triangulated_fraction * nmatches:
            return True, new_map_points
        else:
            return False, new_map_points


    def warp_features(self, points, H):
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)
        warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
        warped_shitomasi_points = warped_points_np.astype(int).reshape(-1,1,2)

        return warped_shitomasi_points

    def create_mask(self, corners):
        """
        Create a binary mask from rectangle bounds [x_min, y_min, x_max, y_max].
        :param shape: Shape of the mask (height, width).
        :param rect: List [x_min, y_min, x_max, y_max].
        :return: Binary mask with the rectangle filled.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if corners is not None:
            for rect in corners:
                values = np.asarray(rect, dtype=np.float64).reshape(-1)
                if values.size != 4 or not np.all(np.isfinite(values)):
                    continue
                x_min, y_min, x_max, y_max = values
                x_min = int(np.clip(np.floor(x_min), 0, self.width - 1))
                y_min = int(np.clip(np.floor(y_min), 0, self.height - 1))
                x_max = int(np.clip(np.ceil(x_max), 0, self.width - 1))
                y_max = int(np.clip(np.ceil(y_max), 0, self.height - 1))
                if x_min >= x_max or y_min >= y_max:
                    continue
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        return mask


    #                 # update the 3d points after triangulation
