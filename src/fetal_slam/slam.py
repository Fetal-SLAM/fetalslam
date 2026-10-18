#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import sys
import cv2
import threading
from time import time
import argparse
import os

from .parameters import Parameter
from .features.detector import Xfeat_Det_Matching
from .frame import CameraFrame
from .keyframe import KeyFrame, KeyFrameTree
from .optimizer import Optimizer
from .visual_reconstruction import VisualReconstruction
from .map_point import MapPoint
from .map import Mapping
from .local_mapping import LocalMapping
from .dense_job import snapshot_dense_scene, run_dense_after_sequence
from .loop_closing import LoopClosing
from .viewer import PangolinViewer
from .workers import BackgroundWorker

import logging as _logging

_logger = _logging.getLogger(__name__)


def evaluate_frame_quality(image, blur_laplacian_min, noise_laplacian_max, contrast_min):
    """Return whether an image is reliable enough for feature tracking."""
    if image is None or image.size == 0:
        return False, {"reason": "empty", "laplacian": 0.0, "contrast": 0.0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    if contrast < contrast_min:
        reason = "low_contrast"
    elif laplacian < blur_laplacian_min:
        reason = "blur"
    elif laplacian > noise_laplacian_max:
        reason = "noise"
    else:
        reason = "ok"
    return reason == "ok", {
        "reason": reason,
        "laplacian": laplacian,
        "contrast": contrast,
    }


class XFeatSlamOffline(Parameter):

    def __init__(self, sys_param, cam_param, data_folder, data_mode="auto", image_subdir="undist_image"):
        super().__init__()
        # setup system paramters
        self.load_sys_param(sys_param, cam_param)
        # set camera matrix and distortion coefficient parameters
        self.mtx, self.dist = self.get_cam_K_distco()
        # set image size
        self.img_width = self.cam_intrinsic.width
        self.img_height = self.cam_intrinsic.height
        self.margin = self.runtime_config['tracking'].image_margin
        self.max_fake_features = self.runtime_config['tracking'].max_fake_features
        self.iou_threshold = self.runtime_config['tracking'].keyframe_iou_threshold
        self.close_loop_iou_threshold = float(self.system_config.loop_iou_threshold)

        
        # setup system state
        self.initial_flag = False
        self.init_cam_frame = None
        self.xfeat_features = None
        self.kToSecond = 1e-9
        # counter for the data
        self.counter = 0

        # load data from file
        self.data_mode = self.resolve_data_mode(data_folder, data_mode)
        self.load_from_folder = self.data_mode == "sensor"
        self.display_image = None
        if self.data_mode == "sensor":
            data_loader_flag, self.data_list, self.time_encoder_map = self.load_multi_sensor_data(data_folder)
            # set data length
            self.data_length = len(self.data_list)

        elif self.data_mode == "npy":
            data_loader_flag, self.images_list, self.timestamp_list = self.load_multi_sensor_data_from_npy(data_folder)
            # set data length
            self.data_length = len(self.images_list)

        else:
            data_loader_flag, self.images_list, self.timestamp_list = self.load_image_sequence_data(data_folder, image_subdir)
            self.data_length = len(self.images_list)

        if not data_loader_flag:
            _logger.warning('%s', "cannot load data from the folder, please check the folder path!")
            sys.exit(-1)

        # set paramters for image undistortion
        self.new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.img_width, self.img_height), 0, (self.img_width, self.img_height))

        # set the mutex lock for threads
        self.mMutexMapUpate = threading.Lock()

        # initial the xfeat
        self.xfeat = Xfeat_Det_Matching(max_kpts=self.tracker.max_kpts, img_margin=self.margin, max_fake_features=self.max_fake_features, settings=self.runtime_config['features'])

        # initial the optimizer
        self.optimizer = Optimizer(self.new_mtx, self.img_width, self.img_height, settings=self.runtime_config['optimizer'])

        # initial the visual reconstruction
        self.vis_recon = VisualReconstruction(
            self.new_mtx,
            None,
            self.img_width,
            self.img_height,
            margin_percent=self.margin,
            min_point_parallax_degrees=self.system_config.triangulation_min_parallax_degrees,
            max_depth_factor=self.system_config.triangulation_max_depth_factor,
            settings=self.runtime_config['reconstruction'],
        )

        # initial the map
        self.map = Mapping()

        # Interactive 3D map, keyframe, and trajectory visualization.
        self.viewer = PangolinViewer(settings=self.runtime_config['viewer'])

        # initial keyframe tree
        self.kf_tree = KeyFrameTree(max_parents=self.runtime_config['tracking'].max_parents, iou_threshold=self.runtime_config['tracking'].parent_repeat_iou, gp_promote_threshold=self.runtime_config['tracking'].grandparent_promote_iou)
        self.kf_tree.set_xfeat(self.xfeat)
        self.kf_tree.set_recon(self.vis_recon)

        # initial localmapping
        self.local_mapping = LocalMapping(self.img_width, self.img_height, max_fake_features=self.max_fake_features, settings=self.runtime_config['local_mapping'])
        self.local_mapping.local_point_reuse_enabled = self.system_config.local_point_reuse_enabled
        self.local_mapping.set_vis_recon(self.vis_recon)
        self.local_mapping.set_xfeat_detector(self.xfeat)
        self.local_mapping.set_optimizer(self.optimizer)
        self.local_mapping.set_KeyFrameTree(self.kf_tree)
        self.local_mapping.set_map(self.map)
        self.local_mapping.configure_triangulation(
            self.system_config.triangulation_min_parallax_degrees,
            self.system_config.triangulation_max_depth_factor,
            self.system_config.triangulation_distance_ratio,
        )

        # initial loopclosing
        self.loop_closing = LoopClosing(self.img_width, self.img_height, self.margin, self.close_loop_iou_threshold, settings=self.runtime_config['loop_closing'])
        self.loop_closing.set_xfeat(self.xfeat)
        self.loop_closing.set_optimizer(self.optimizer)
        self.loop_closing.set_vis_recon(self.vis_recon)
        self.loop_closing.set_KeyFrameTree(self.kf_tree)
        self.loop_closing.set_map(self.map)
        self.loop_closing.configure_place_recognition(
            self.system_config.loop_min_frame_gap,
            self.system_config.loop_cooldown_frames,
            self.system_config.place_candidate_count,
            self.system_config.loop_confirmation_count,
            self.system_config.relocalization_iou_threshold,
        )
        self.loop_closing.configure_local_map_recovery(
            enabled=self.system_config.local_map_recovery_enabled,
            interval=self.system_config.local_map_recovery_interval,
            min_frame_gap=self.system_config.local_map_recovery_min_frame_gap,
            candidate_count=self.system_config.local_map_recovery_candidate_count,
            iou_threshold=self.system_config.local_map_recovery_iou_threshold,
            min_inliers=self.system_config.local_map_recovery_min_inliers,
            min_observations=self.system_config.local_map_recovery_min_observations,
            max_points=self.system_config.local_map_recovery_max_points,
            reprojection_error=self.system_config.local_map_recovery_reprojection_error,
            max_pose_jump_depth_factor=self.system_config.local_map_recovery_max_pose_jump_depth_factor,
        )

        self.local_mapping_worker = BackgroundWorker(
            "LocalMapping", lambda _: self.local_mapping.run()
        )
        self.loop_closing_worker = BackgroundWorker(
            "LoopClosing", self.run_loop_closing_worker
        )
        self.local_mapping_worker.start()
        self.loop_closing_worker.start()
        self.workers_stopped = False
        self.cached_keyframe_count = 0
        self.frames_while_mapping_busy = 0
        self.tracking_state = "INITIALIZING"
        self.tracking_lost_count = 0
        self.skipped_bad_frames = 0
        self.relocalization_count = 0
        self.local_map_recovery_count = 0
        self.local_map_recovered_points = 0
        self.last_frame_quality = {"reason": "unknown", "laplacian": 0.0, "contrast": 0.0}


        # Set the cv2 image window
        self.window_name = "Real-time matching."
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.img_width, self.img_height)

        # create two dictionary for data recording
        self.cam_traj_record = {}
        self.map_points_record = None
        self.ee_pose_record = {}

        self.time_list_feature_matching = []
        self.time_list_pose_estimation = []
        self.time_list_local_mapping = []

        
    def resolve_data_mode(self, data_folder, data_mode):
        if data_mode != "auto":
            return data_mode
        if os.path.isdir(os.path.join(data_folder, "sensor_data")):
            return "sensor"
        if os.path.exists(os.path.join(data_folder, "xfeat_outputsave_ref_frames.npy")):
            return "npy"
        return "image-sequence"

    def draw_slam_overlay(self, image):
        view = image.copy()
        keyframes = self.cached_keyframe_count
        map_points = len(self.map.get_all_map_points())
        frame_id = self.cur_cam_frame.get_frame_id() if hasattr(self, "cur_cam_frame") else -1
        ref_id = self.last_parent_key_frame.mnFrameId if hasattr(self, "last_parent_key_frame") else -1
        xfeat_state = getattr(self, "xfeat_flag", None)
        lines = [
            f"frame {frame_id} | ref {ref_id}",
            f"{self.tracking_state} | xfeat {xfeat_state} | lost {self.tracking_lost_count}",
            f"quality {self.last_frame_quality['reason']} | sharp {self.last_frame_quality['laplacian']:.1f}",
            f"KF {keyframes} | MP {map_points}",
            f"skipped {self.skipped_bad_frames} | relocalized {self.relocalization_count}",
            f"local recovery {self.local_map_recovery_count} | +MP {self.local_map_recovered_points}",
            f"LM {'busy' if self.local_mapping_worker.busy else 'idle'} | "
            f"LC {'busy' if self.loop_closing_worker.busy else 'idle'}",
        ]
        for idx, line in enumerate(lines):
            y = 24 + idx * 24
            cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        return view

    @property
    def background_busy(self):
        return self.local_mapping_worker.busy or self.loop_closing_worker.busy

    def consume_background_results(self, track_current=True):
        reference_tracking_result = None
        for result in self.local_mapping_worker.poll():
            self.time_list_local_mapping.append(result.elapsed)
            if result.error is not None:
                raise RuntimeError("Local mapping worker failed") from result.error

            mapping_result = result.value
            if mapping_result["new_parent"]:
                reference_tracking_result = self.adopt_new_parent(track_current)
                self.kf_tree.new_parent_flag = False
                if not self.loop_closing_worker.submit(mapping_result["new_parent_name"]):
                    raise RuntimeError("Loop closing worker rejected a completed map update")

        for result in self.loop_closing_worker.poll():
            if result.error is not None:
                raise RuntimeError("Loop closing worker failed") from result.error

            worker_value = result.value
            recovery = worker_value["recovery"]
            if recovery["accepted"]:
                self.local_map_recovery_count += 1
                self.local_map_recovered_points += recovery["added_points"]
                _logger.debug('%s', "local map recovery accepted: "
                    f"grandparent={recovery['candidate_id']}, "
                    f"inliers={recovery['inliers']}, added={recovery['added_points']}")

            optimized_pose, _, close_loop_candidate = worker_value["loop"]
            if optimized_pose is not None:
                _logger.info('%s', "loop closing success!")
                self.kf_tree.add_parent_node_after_close_loop(close_loop_candidate)
                self.viewer.set_optimized_trajectory(optimized_pose)
                reference_tracking_result = self.adopt_new_parent(track_current)

            elif recovery["accepted"]:
                reference_tracking_result = self.adopt_new_parent(track_current)

        if not self.background_busy:
            self.cached_keyframe_count = len(
                self.kf_tree.get_all_keyframes_for_loopclosing()
            )
        return reference_tracking_result

    def run_loop_closing_worker(self, parent_name):
        # Recovery and loop correction rewrite shared poses and observations atomically.
        with self.mMutexMapUpate:
            if isinstance(parent_name, KeyFrame):
                probe = parent_name
                probe.set_xfeat_features(self.xfeat.detect(probe.image))
                _logger.debug('%s', f"periodic global loop query at frame {probe.mnFrameId}")
                return {
                    "recovery": self.loop_closing._local_recovery_result("global-query"),
                    "loop": self.loop_closing.run(force=True, query_keyframe=probe),
                }
            recovery = self.loop_closing.recover_local_map(parent_name)
            return {
                "recovery": recovery,
                "loop": self.loop_closing.run(force=True),
            }

    def maybe_submit_loop_probe(self):
        interval = int(self.system_config.loop_check_interval)
        if interval <= 0 or self.background_busy:
            return False
        frame_id = self.cur_cam_frame.get_frame_id()
        last_query = self.loop_closing.last_loop_query_frame_id
        if last_query is not None and frame_id - last_query < interval:
            return False
        if frame_id - self.loop_closing.last_close_loop_keyframe_id < self.system_config.loop_cooldown_frames:
            return False
        # A query reserves an ID but does not enter the map unless confirmed.
        probe = KeyFrame(self.cur_cam_frame)
        probe.KeysUn_dict = {key: value.copy() for key, value in probe.KeysUn_dict.items()}
        probe.mvKeysUn = probe.mvKeysUn.copy()
        return self.loop_closing_worker.submit(probe)

    def adopt_new_parent(self, track_current=True):
        new_parent = self.kf_tree.newest_parent_name
        parent_node = self.kf_tree.parents[new_parent]["node"]
        _logger.debug('%s %s', "new parent feature length before: ", parent_node.kf.N)
        _logger.debug('%s %s', "new_parent: ", parent_node.name)
        self.last_parent_key_frame = parent_node.get_kf()
        self.xfeat.update_ref_frame(self.last_parent_key_frame.image)
        self.last_parent_key_frame.update_features_and_mappoints()
        self.cur_cam_frame.reference_kf = self.last_parent_key_frame
        _logger.debug('%s %s', "new parent feature length after: ", len(self.last_parent_key_frame.get_mvKeysUn()))
        if not track_current:
            return None
        tracking_result = self.update_and_track_reference_key_frame()
        # Keep feature IDs aligned with the promoted parent even when pose
        # optimization rejects this transition frame.
        self.last_cam_frame = self.cur_cam_frame
        return tracking_result

    def stop_workers(self):
        if self.workers_stopped:
            return
        self.local_mapping_worker.stop()
        self.consume_background_results(track_current=False)
        self.loop_closing_worker.stop()
        self.consume_background_results(track_current=False)
        self.workers_stopped = True
        _logger.debug('%s', "worker timing: "
            f"local={self.local_mapping_worker.jobs_completed} jobs/"
            f"{self.local_mapping_worker.total_elapsed:.3f}s, "
            f"loop={self.loop_closing_worker.jobs_completed} jobs/"
            f"{self.loop_closing_worker.total_elapsed:.3f}s")
        _logger.debug('%s', "thread overlap: "
            f"{self.frames_while_mapping_busy} frames tracked while local mapping was scheduled")
        _logger.debug('%s', "tracking recovery: "
            f"skipped={self.skipped_bad_frames}, relocalized={self.relocalization_count}, "
            f"local_recovery={self.local_map_recovery_count}/"
            f"{self.local_map_recovered_points} points, final_state={self.tracking_state}")


    def run(self, data_folder, max_frames=None):
        t0 = time()
        processed_frames = 0
        user_quit = False
        if self.load_from_folder:
            # help the dataload the prepare the encoder data
            image_flag = False
            for data in self.data_list:
                time_str, sensor_type, *_ = data
                timestamp = float(time_str) * self.kToSecond
                if sensor_type == "stereo":
                    img_file = f"{data_folder}/undist_image/{time_str}.png"
                    image = cv2.imread(img_file)
                    image_flag = True
                    self.tracking(image, timestamp)
                    processed_frames += 1
                    cv2.imshow(self.window_name, self.display_image if self.display_image is not None else image)
                    key = cv2.waitKey(1)  & 0xFF
                    if key in (ord('q'), 27) or self.viewer.closed:
                        user_quit = True
                        break
                    if max_frames is not None and processed_frames >= max_frames:
                        break
                if sensor_type == "encoder" and image_flag:
                    img_file = f"{data_folder}/undist_image/{time_str}.png"
                    if time_str in self.time_encoder_map:
                        encoder_data = self.time_encoder_map[time_str].split(',')
                        x = float(encoder_data[0])
                        y = float(encoder_data[1])
                        z = float(encoder_data[2])
                        rx = float(encoder_data[3])
                        ry = float(encoder_data[4])
                        rz = float(encoder_data[5])
                        em = [x,y,z,rx,ry,rz]
                        frame_id = self.cur_cam_frame.get_frame_id()
                        self.ee_pose_record[frame_id] = em.copy()
                        
                    image_flag = False
        
        else:
            for k, data in enumerate(self.images_list):
                image = data
                timestamp = self.timestamp_list[k]
                self.tracking(image, timestamp)
                processed_frames += 1

                cv2.imshow(self.window_name, self.display_image if self.display_image is not None else image)
                key = cv2.waitKey(1)  & 0xFF
                if key in (ord('q'), 27) or self.viewer.closed:
                    user_quit = True
                    break
                if max_frames is not None and processed_frames >= max_frames:
                    break

        
        self.stop_workers()
        t1 = time() - t0
        FPS = processed_frames/t1

        _logger.info('%s', f"total time: {t1}s with FPS {FPS} for {processed_frames} frames")


        cv2.destroyAllWindows()
        self.viewer.close()
        if not user_quit and max_frames is None and self.initial_flag:
            frames = self.kf_tree.get_all_keyframes_for_loopclosing()
            frames = dict(frames)
            for frame in self.map.get_all_key_frames():
                frames[frame.mnId] = frame
            for frame in self.kf_tree.get_all_temp_keyframes():
                frames[frame.mnId] = frame
            run_dense_after_sequence(
                lambda: snapshot_dense_scene(
                    frames, self.map.get_all_map_points(), self.new_mtx,
                    **self.dense_config), self.dense_config)


        # # save the data
        # import pickle


    def tracking(self, image, timestamp):
        self.cur_cam_frame = CameraFrame(image, timestamp, self.new_mtx, None, roi_margin=self.margin)
        with self.mMutexMapUpate:
            if self.local_mapping_worker.busy:
                self.frames_while_mapping_busy += 1
            quality_ok, self.last_frame_quality = evaluate_frame_quality(
                image,
                self.system_config.blur_laplacian_min,
                self.system_config.noise_laplacian_max,
                self.system_config.image_contrast_min,
            )
            reference_tracking_result = self.consume_background_results(track_current=quality_ok)

            if not quality_ok:
                self.skipped_bad_frames += 1
                self.tracking_state = f"SKIP_{self.last_frame_quality['reason'].upper()}"
                if self.initial_flag:
                    self.cur_cam_frame.mTcw = self.last_cam_frame.get_pose().copy()
                    self.cur_cam_frame.reference_kf = self.last_parent_key_frame
            elif not self.initial_flag:
                self.tracking_state = "INITIALIZING"
                self.mono_initialization(image, timestamp)
                if self.initial_flag:
                    self.tracking_state = "TRACKING"
            else:
                if reference_tracking_result is None:
                    bOK, iou_cur_ref = self.track_reference_key_frame()
                else:
                    bOK, iou_cur_ref = reference_tracking_result

                if not bOK:
                    self.tracking_lost_count += 1
                    self.tracking_state = "LOST"
                    retry_after = int(self.system_config.relocalization_after_failures)
                    retry_interval = max(1, int(self.system_config.relocalization_interval))
                    should_relocalize = (
                        self.tracking_lost_count >= retry_after
                        and (self.tracking_lost_count - retry_after) % retry_interval == 0
                        and not self.background_busy
                    )
                    if should_relocalize:
                        bOK, iou_cur_ref = self.try_relocalization()

                if bOK:
                    if self.tracking_state != "RELOCALIZED":
                        self.tracking_state = "TRACKING"
                    self.tracking_lost_count = 0
                    if self.need_new_keyframe(iou_cur_ref) and not self.background_busy:
                        _logger.debug('%s', f"{self.cur_cam_frame.frame_id} with iou {iou_cur_ref} to ref frame {self.last_parent_key_frame.mnFrameId}.")

                        child_added = self.create_new_keyframe()
                        if child_added:
                            if not self.local_mapping_worker.submit(
                                self.cur_cam_frame.get_frame_id()
                            ):
                                raise RuntimeError("Local mapping worker rejected a new keyframe")
                            
                    KF_ini_mvpMapPoints = self.cur_cam_frame.mvpMapPoints
                    cur_cam_pose = self.cur_cam_frame.get_pose()
                    frame_id = self.cur_cam_frame.get_frame_id()
                    self.record_cam_traj(cur_cam_pose, self.cur_cam_frame.image ,frame_id)
                    self.last_cam_frame = self.cur_cam_frame
                    self.maybe_submit_loop_probe()
                else:
                    # Preserve the last valid pose while later frames continue.
                    self.cur_cam_frame.mTcw = self.last_cam_frame.get_pose().copy()
                    self.cur_cam_frame.reference_kf = self.last_parent_key_frame


            pub_img = image.copy()
            cur_feature = list(self.cur_cam_frame.get_KeysUn_dict().values())
            if len(cur_feature) > 0:
                for pt in cur_feature:
                    cv2.circle(pub_img, (int(pt[0]), int(pt[1])), 2, (0,255,0), -1)

            pub_img = self.draw_slam_overlay(pub_img)
            self.display_image = pub_img
            viewer_tree = None if self.background_busy else self.kf_tree
            viewer_pose = self.cur_cam_frame.get_pose() if self.initial_flag else None
            self.viewer.render(self.map, viewer_pose, viewer_tree)
            
            return True

    def try_relocalization(self):
        candidate, _, candidate_iou = self.loop_closing.find_relocalization_candidate(
            self.cur_cam_frame.image,
            self.cur_cam_frame.get_frame_id(),
        )
        if candidate is None:
            _logger.warning('%s', f"relocalization failed at frame {self.cur_cam_frame.get_frame_id()}")
            return False, None

        previous_parent = self.last_parent_key_frame
        self.last_parent_key_frame = candidate
        self.xfeat.update_ref_frame(candidate.image)
        self.cur_cam_frame.reference_kf = candidate
        tracking_ok, tracking_iou = self.update_and_track_reference_key_frame()
        if not tracking_ok:
            self.last_parent_key_frame = previous_parent
            self.xfeat.update_ref_frame(previous_parent.image)
            self.cur_cam_frame.reference_kf = previous_parent
            return False, None

        self.kf_tree.add_parent(candidate)
        self.relocalization_count += 1
        self.tracking_state = "RELOCALIZED"
        _logger.debug('%s', f"relocalized frame {self.cur_cam_frame.get_frame_id()} to "
            f"keyframe {candidate.mnFrameId} (candidate iou {candidate_iou:.3f})")
        return True, tracking_iou

    def mono_initialization(self, image, timestamp):
        if self.init_cam_frame is None:
            self.init_cam_frame = CameraFrame(image, timestamp, self.new_mtx, None, roi_margin=self.margin)
            self.init_cam_frame.setpose(np.identity(3),np.array([[0.0],[0.0],[0.0]]))
            init_xfeat_flag, corner_features, xfeat_features = self.xfeat.init_detect(image)

            if init_xfeat_flag is False:
                _logger.debug('%s', "[mono init] init_xfeat_flag is false!")
                return
            else:
                _logger.debug('%s', "[mono init] init_xfeat_flag is true!")
                self.init_cam_frame.set_undist_features(corner_features, np.identity(3))
                self.ini_undist_features = corner_features.copy()

                return
        else:
            xfeat_flag, H, _= self.xfeat.detect_and_match(image)
            
            if xfeat_flag:
                self.homography_matrix = H
                # use homography matrix to 
                current_undist_features = self.xfeat.warp_shitomasi_features(self.ini_undist_features, H)


                recon_flag, R21, t21, vP3D, vbTriangulated = self.vis_recon.ReconstructH(self.ini_undist_features, current_undist_features, H)

                if recon_flag:
                    # Tcw is world respect to camera, as the mappoints are fixed and the camera is moving
                    self.cur_cam_frame.setpose(R21, t21)
                    self.cur_cam_frame.set_undist_features(current_undist_features, H)
                    self.cur_cam_frame.set_map_points(vP3D)
                    iou_cur_ref = self.cur_cam_frame.warp_roi_with_ref_kf(H)
                    _logger.debug('%s %s', "iou_cur_ref: ", iou_cur_ref)
                    self.cur_cam_frame.set_vbTriangulated(vbTriangulated)
                    self.init_cam_frame.set_vbTriangulated(vbTriangulated)

                    self.initial_flag = self.create_initial_map_monocular()
                    if not self.initial_flag:
                        # Retry later from a fresh reference without retaining
                        # provisional keyframes or map points.
                        self.init_cam_frame = None

                    
                else:
                    _logger.debug('%s', "initial fails, please try again")
                
            return


    def create_initial_map_monocular(self):
        # create keyframes
        KF_ini = KeyFrame(self.init_cam_frame)
        KF_cur = KeyFrame(self.cur_cam_frame)
        self.optimizer.set_init_frame_id(KF_ini.mnId)

        KF_ini.set_xfeat_features(self.xfeat.detect(KF_ini.image))
        KF_cur.set_xfeat_features(self.xfeat.detect(KF_cur.image))

        raw_map_points = self.cur_cam_frame.get_map_points()
        valid_indices = [
            index
            for index, triangulated in enumerate(KF_ini.vbTriangulated)
            if triangulated
            and raw_map_points[index] is not None
            and np.all(np.isfinite(raw_map_points[index]))
        ]
        valid_depths = np.array(
            [float(np.asarray(raw_map_points[index]).reshape(3)[2]) for index in valid_indices]
        )
        medianDepth = float(np.median(valid_depths)) if valid_depths.size else 0.0
        _logger.debug('%s %s', "medianDepth: ", medianDepth)

        if (
            not np.isfinite(medianDepth)
            or medianDepth <= 0
            or len(valid_indices) < int(self.system_config.initial_min_map_points)
        ):
            _logger.debug('%s', "Wrong initialization, retrying with a new reference "
                f"({len(valid_indices)} valid map points).")
            return False
        invMedianDepth = 1.0 / medianDepth


        # step 6. Normalize the transformation between two frames to an average depth of 1
        Tc2w = KF_cur.get_pose()
        Tc2w[:3, 3] = Tc2w[:3, 3] * invMedianDepth
        KF_cur.set_pose(Tc2w)
        # Keep the live frame and viewer in the same normalized map scale as
        # the keyframe. Otherwise the raw initialization translation becomes
        # a single distant point in the dark live trajectory.
        self.cur_cam_frame.setpose(Tc2w[:3, :3], Tc2w[:3, 3])
        self.viewer.set_initial_trajectory((KF_ini.get_pose(), KF_cur.get_pose()))


        # Create persistent points only after validation. Map-point IDs and
        # feature-array indices are intentionally separate.
        for feature_index in valid_indices:
            world_pos = np.asarray(raw_map_points[feature_index]).reshape(3, 1)
            pMP = MapPoint(world_pos * invMedianDepth, KF_ini)
            KF_ini.add_map_point(pMP, pMP.mnId, feature_index)
            KF_cur.add_map_point(pMP, pMP.mnId, feature_index)
            pMP.add_observation(KF_ini.mnId)
            pMP.add_observation(KF_cur.mnId)
            feature_2d = KF_ini.mvKeysUn[feature_index][0]
            pMP.set_map_point_color(KF_ini.get_feature_color(feature_2d))

        # Rejected initialization points leave holes in the original feature
        # arrays. Compact both keyframes so array row i and dictionary item i
        # continue to describe the same 2D-to-3D correspondence.
        KF_ini.compact_features_to_map_points()
        KF_cur.compact_features_to_map_points()

        # Commit only after geometry validation and scale normalization.
        _logger.debug('%s', "add key frame")
        self.map.add_key_frame(KF_ini)
        self.map.add_key_frame(KF_cur)
        self.map.add_map_points(KF_ini.get_mvpMapPoints().values())

        # step 8. Insert the keyframe into the local map and update the normalized pose and local map points


        self.kf_tree.add_parent(KF_ini)

        self.kf_tree.add_grand_parent(KF_ini)
        self.cached_keyframe_count = len(
            self.kf_tree.get_all_keyframes_for_loopclosing()
        )


        KF_ini.mTriangle_seed = False
        self.last_parent_key_frame = KF_ini


        self.mvlocalMapPoints = self.map.get_all_map_points()
        self.reference_Kf = KF_ini


        self.cur_cam_frame.reference_kf = KF_ini
        self.cur_cam_frame.mvpMapPoints = KF_ini.get_mvpMapPoints()
        self.cur_cam_frame.KeysUn_dict = KF_cur.get_KeysUn_dict()

        self.last_cam_frame = self.cur_cam_frame


        self.map.set_reference_map_points(self.mvlocalMapPoints)


        self.map.mvp_key_frame_origins.append(KF_ini)


        return True


    def track_reference_key_frame(self):
        xfeat_flag, H, good_matches = self.xfeat.detect_and_match(self.cur_cam_frame.image)
        self.xfeat_flag = xfeat_flag

        if xfeat_flag is False:
            return False, None
        
        self.good_matches = good_matches
        self.homography_matrix = H
        undist_features = self.last_parent_key_frame.get_mvKeysUn()
        # use homography matrix to 
        current_undist_features = self.xfeat.warp_shitomasi_features(undist_features, H)


        self.cur_cam_frame.set_undist_features(current_undist_features, H)
        iou_cur_ref = self.cur_cam_frame.warp_roi_with_ref_kf(H)


        # step 3: Use the previous frame's pose as the initial value of the current frame's pose
        self.cur_cam_frame.mTcw = self.last_cam_frame.get_pose()
        self.cur_cam_frame.vbTriangulated = self.last_parent_key_frame.get_vbTriangulated()
        self.cur_cam_frame.mvpMapPoints = self.last_parent_key_frame.get_mvpMapPoints()
        feature_count = len(self.cur_cam_frame.mvpMapPoints)
        self.cur_cam_frame.vbTriangulated = (
            list(self.cur_cam_frame.vbTriangulated) + [False] * feature_count
        )[:feature_count]
        self.cur_cam_frame.mvbOutlier = [False] * feature_count
        self.cur_cam_frame.KeysUn_dict = self.last_parent_key_frame.get_KeysUn_dict()

        self.cur_cam_frame.update_undist_features_in_KeysUn_dict()
        # step 4:
        optimized_flag, optimized_frame = self.optimizer.pose_optimization2(self.cur_cam_frame)


        if optimized_flag > 10:
            self.cur_cam_frame = optimized_frame
            
            return True, iou_cur_ref
        else:
            return False, iou_cur_ref
        
    def update_and_track_reference_key_frame(self):
        # step 1. store the descriptor and features in tensor
        # to do

        # step 2. match features 
        xfeat_flag, H, good_matches = self.xfeat.detect_and_match(self.cur_cam_frame.image)
        self.xfeat_flag = xfeat_flag

        if xfeat_flag is False:
            return False, None
        
        self.homography_matrix = H

        undist_features = self.last_parent_key_frame.get_mvKeysUn()

        # use homography matrix to 
        current_undist_features = self.xfeat.warp_shitomasi_features(undist_features, H)

        self.cur_cam_frame.set_undist_features(current_undist_features, H)
        iou_cur_ref = self.cur_cam_frame.warp_roi_with_ref_kf(H)
        
        # step 3: Use the previous frame's pose as the initial value of the current frame's pose
        self.cur_cam_frame.mTcw = self.last_parent_key_frame.get_pose()
        self.cur_cam_frame.vbTriangulated = self.last_parent_key_frame.get_vbTriangulated()
        self.cur_cam_frame.mvpMapPoints = self.last_parent_key_frame.get_mvpMapPoints()
        feature_count = len(self.cur_cam_frame.mvpMapPoints)
        self.cur_cam_frame.vbTriangulated = (
            list(self.cur_cam_frame.vbTriangulated) + [False] * feature_count
        )[:feature_count]
        self.cur_cam_frame.mvbOutlier = [False] * feature_count
        self.cur_cam_frame.KeysUn_dict = self.last_parent_key_frame.get_KeysUn_dict()

        self.cur_cam_frame.update_undist_features_in_KeysUn_dict()
        # step 4:
        optimized_flag, optimized_frame = self.optimizer.pose_optimization2(self.cur_cam_frame)

        if optimized_flag > 10:
            self.cur_cam_frame = optimized_frame
            
            return True, iou_cur_ref
        else:
            return False, iou_cur_ref

        
    def need_new_keyframe(self, iou_cur_ref):
        # step 1: if iou < 0.75, skip this frame
        a = iou_cur_ref < self.iou_threshold
        if a:
            return True
        else:
            return False
        
    def create_new_keyframe(self):
        new_child_keyframe = KeyFrame(self.cur_cam_frame)
        new_child_keyframe.set_xfeat_features(self.xfeat.detect(new_child_keyframe.image))
        
        return self.kf_tree.add_child_to_parent(
            self.kf_tree.newest_parent_name,
            new_child_keyframe,
            self.homography_matrix,
        )
                    

    def record_cam_traj(self, cam_traj, image, id):
        self.cam_traj_record[id] = {"pose":cam_traj.copy(),"image":image.copy()}


def main(sys_param, cam_param, data_folder, data_mode="auto", image_subdir="undist_image", max_frames=None):
    xSLAM = XFeatSlamOffline(sys_param, cam_param, data_folder, data_mode=data_mode, image_subdir=image_subdir)
    try:
        xSLAM.run(data_folder, max_frames=max_frames)
    except KeyboardInterrupt:
        _logger.info('%s', "SLAM stopped by user.")
    finally:
        xSLAM.stop_workers()
        cv2.destroyAllWindows()
        xSLAM.viewer.close()


def cli():
    parser = argparse.ArgumentParser(
        prog="fetal-slam",
        description="Run full XFeat SLAM with OpenCV and Pangolin viewers.",
    )
    parser.add_argument("sys_param", help="System YAML, for example config/param/vo_fast.yaml")
    parser.add_argument("cam_param", help="Camera YAML, for example config/param/feto_cam_exvivo.yaml")
    parser.add_argument("data_folder", help="Dataset folder.")
    parser.add_argument(
        "--data-mode",
        choices=["auto", "sensor", "npy", "image-sequence"],
        default="auto",
        help="Input format. auto chooses sensor_data, npy cache, or image sequence.",
    )
    parser.add_argument(
        "--image-subdir",
        default="undist_image",
        help="Image folder inside data_folder when --data-mode image-sequence is used.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many images. Useful for repeatable performance tests.",
    )
    args = parser.parse_args()
    main(
        args.sys_param,
        args.cam_param,
        args.data_folder,
        data_mode=args.data_mode,
        image_subdir=args.image_subdir,
        max_frames=args.max_frames,
    )


if __name__ == '__main__':
    cli()
