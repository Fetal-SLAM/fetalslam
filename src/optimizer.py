import g2o
import numpy as np

from .keyframe import KeyFrameTree

import logging as _logging

_logger = _logging.getLogger(__name__)


from .runtime_config import runtime_settings


class Optimizer():

    def __init__(self, K, width, height, settings=None):
        self.settings = settings or runtime_settings("optimizer")
        self.mK = K
        self.focal_length = K[0,0]
        self.cx = K[0,2]
        self.cy = K[1,2]
        self.width = width
        self.height = height

        self.init_frame_id = 0


    def set_init_frame_id(self, init_frame_id):
        self.init_frame_id = init_frame_id

    @staticmethod
    def allocate_vertex_ids(existing_ids, domain_ids):
        """Assign compact g2o IDs without relying on sparse SLAM object IDs."""
        existing_ids = list(existing_ids)
        next_id = max(existing_ids, default=-1) + 1
        return {
            domain_id: next_id + offset
            for offset, domain_id in enumerate(domain_ids)
        }

    @staticmethod
    def select_fixed_frame_id(frame_ids, requested_id=0):
        frame_ids = list(frame_ids)
        if requested_id in frame_ids:
            return requested_id
        return min(frame_ids, default=None)

    @staticmethod
    def pose_observations(frame):
        """Return map points paired with their ID-keyed image observations."""
        observations = []
        feature_by_map_point = frame.KeysUn_dict
        for position, (map_point_id, map_point) in enumerate(frame.mvpMapPoints.items()):
            feature = feature_by_map_point.get(map_point_id)
            if map_point is None or feature is None:
                continue
            observations.append(
                (
                    position,
                    map_point_id,
                    map_point,
                    np.asarray(feature, dtype=float).reshape(2).copy(),
                )
            )
        return observations


    #     # Optimize

    #     # Recover optimized data


    def isometry3d_to_numpy(self, isometry):
        """
        Converts a g2o.Isometry3d object to a 4x4 numpy array.
        :param isometry: g2o.Isometry3d object.
        :return: 4x4 numpy array representing the transformation matrix.
        """
        # Extract rotation (3x3) and translation (3x1)
        rotation_owner = isometry.rotation()
        rotation = np.array(rotation_owner.matrix(), copy=True)
        translation = np.array(isometry.translation(), copy=True)

        # Construct the 4x4 homogeneous transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation

        return transformation_matrix
    
    def pose_optimization2(self, frame):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        n_initial_correspondences = 0

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(0)
        pose = g2o.Isometry3d(frame.get_pose())


        v_se3.set_estimate(g2o.SE3Quat(pose.rotation(), pose.translation()))
        v_se3.set_fixed(False)
        optimizer.add_vertex(v_se3)

        # Prepare data structures for edges
        edges_mono = []
        index_edges_mono = []
        edges_stereo = []
        index_edges_stereo = []

        delta_mono = np.sqrt(self.settings.mono_chi2)
        delta_stereo = np.sqrt(self.settings.stereo_chi2)

        for idx, key, map_point, mv_keys_un in self.pose_observations(frame):
            if 0 <= mv_keys_un[0] < self.width and 0 <= mv_keys_un[1] < self.height:
                n_initial_correspondences += 1
                frame.mvbOutlier[idx] = False

                obs = np.array([mv_keys_un[0], mv_keys_un[1]])
                edge = g2o.EdgeSE3ProjectXYZOnlyPose()
                edge.set_vertex(0, v_se3)
                edge.set_measurement(obs)
                inv_sigma2 = 1.0
                edge.set_information(np.identity(2) * inv_sigma2)

                edge.fx = frame.fx
                edge.fy = frame.fy
                edge.cx = frame.cx
                edge.cy = frame.cy
                edge.Xw = map_point.get_world_pos().reshape(-1)


                robust_kernel = g2o.RobustKernelHuber()
                edge.set_robust_kernel(robust_kernel)
                robust_kernel.set_delta(delta_mono)

                
                optimizer.add_edge(edge)
                edges_mono.append(edge)
                index_edges_mono.append(idx)
        
        optimizer.initialize_optimization()
        optimizer.optimize(self.settings.iterations)

        n_inliers = 0
        for edge, feature_position in zip(edges_mono, index_edges_mono):
            edge.compute_error()
            is_outlier = (
                not edge.is_depth_positive()
                or not np.isfinite(edge.chi2())
                or edge.chi2() > self.settings.mono_chi2
            )
            frame.mvbOutlier[feature_position] = is_outlier
            if not is_outlier:
                n_inliers += 1

        optimized_pose = v_se3.estimate()
        Rcw = optimized_pose.rotation().matrix()
        tcw = optimized_pose.translation()
        frame.setpose(Rcw, tcw)

        return n_inliers, frame


    def local_bundle_adjustment(self, frame_tree:KeyFrameTree, loop_kf = 0):
        # get frames from KeyframeTree
        frames = {}
        map_points_and_ids = []
        for kf in frame_tree.get_all_temp_keyframes():
            frames[kf.mnId] = kf
            map_points_and_ids.append(kf.get_mvpMapPoints())

            
        merged_dict = {}
        for d in map_points_and_ids:
            merged_dict.update(d)

        input_poses = {key: frame.get_pose().copy() for key, frame in frames.items()}
        input_points = {key: point.get_world_pos().reshape(3).copy() for key, point in merged_dict.items()}
        optimized_frames, optimized_map_points = self.bundle_adjustment(frames, merged_dict)
        report = self.validate_local_ba_geometry(
            frames, merged_dict, input_poses, input_points,
            optimized_frames, optimized_map_points,
        )
        self.last_local_ba_validation = report
        if not report["accepted"]:
            self.local_ba_rejections = getattr(self, "local_ba_rejections", 0) + 1
            _logger.warning('%s', f"local BA rejected: {report}; retaining input poses and points")
            return input_poses, input_points

        return optimized_frames, optimized_map_points

    def validate_local_ba_geometry(self, frames, points, before_poses, before_points,
                                   after_poses, after_points):
        """Validate a local BA transaction in camera-relative, scale-free units.

        This is a rejection gate, not a rescaling operation. Initialization has
        its own normalization and does not use this established-map gate.
        """
        def reject(reason, **metrics):
            return {"accepted": False, "reason": reason, **metrics}

        if not all(np.all(np.isfinite(value)) for value in
                   list(after_poses.values()) + list(after_points.values())):
            return reject("non-finite geometry")
        if set(after_poses) != set(before_poses):
            return reject("incomplete camera solution")

        min_depth_ratio = 1.0
        checked_frames = 0
        flattened_frames = []
        for frame_id, frame in frames.items():
            ids = []
            for point_id, point in points.items():
                if point.is_bad() or frame_id not in point.get_observations():
                    continue
                feature = frame.get_KeysUn_dict_feature(point_id)
                if feature is None:
                    continue
                feature = np.asarray(feature).reshape(2)
                if not np.all(np.isfinite(feature)):
                    continue
                if 0 <= feature[0] < self.width and 0 <= feature[1] < self.height:
                    ids.append(point_id)
            if len(ids) < self.settings.guard_min_points:
                continue
            old_world = np.asarray([before_points[key] for key in ids]).reshape(-1, 3)
            new_world = np.asarray([after_points.get(key, before_points[key]) for key in ids]).reshape(-1, 3)
            old_pose, new_pose = before_poses[frame_id], after_poses[frame_id]
            old_camera = old_world @ old_pose[:3, :3].T + old_pose[:3, 3]
            new_camera = new_world @ new_pose[:3, :3].T + new_pose[:3, 3]
            visible = np.isfinite(old_camera).all(axis=1) & (old_camera[:, 2] > 0)
            old_camera, new_camera = old_camera[visible], new_camera[visible]
            if len(old_camera) < self.settings.guard_min_points:
                continue
            checked_frames += 1
            old_depth, new_depth = old_camera[:, 2], new_camera[:, 2]
            ratios = new_depth / old_depth
            median_ratio = float(np.median(ratios))
            min_depth_ratio = min(min_depth_ratio, median_ratio)
            metrics = {"frame_id": frame_id, "median_depth_ratio": median_ratio}
            if np.mean(new_depth <= 0) > self.settings.guard_behind_fraction:
                return reject("points moved behind camera", **metrics)
            if median_ratio < self.settings.guard_median_depth_ratio:
                return reject("median depth collapse", **metrics)
            if np.mean(ratios < self.settings.guard_partial_depth_ratio) > self.settings.guard_partial_fraction:
                return reject("partial depth collapse", **metrics)

            # Depth alone misses flattening at nearly unchanged mean distance.
            old_span = float(np.ptp(np.percentile(old_depth, [10, 90])))
            new_span = float(np.ptp(np.percentile(new_depth, [10, 90])))
            if old_span > self.settings.guard_min_depth_span * np.median(old_depth) and new_span < self.settings.guard_depth_span_ratio * old_span:
                flattened_frames.append({"depth_span_ratio": new_span / old_span, **metrics})
            old_radius = float(np.median(np.linalg.norm(old_camera - np.median(old_camera, axis=0), axis=1)))
            new_radius = float(np.median(np.linalg.norm(new_camera - np.median(new_camera, axis=0), axis=1)))
            if old_radius > 0 and new_radius < self.settings.guard_radius_ratio * old_radius:
                return reject("local point cloud contracted", extent_ratio=new_radius / old_radius, **metrics)

        # One old view can become fronto-parallel after a legitimate pose update.
        # Require window-level support for flattening, not a persistent veto by
        # a single keyframe with an unreliable initial depth distribution.
        if len(flattened_frames) >= max(self.settings.guard_min_flattened_frames, int(np.ceil(self.settings.guard_flattened_fraction * checked_frames))):
            worst = min(flattened_frames, key=lambda item: item["depth_span_ratio"])
            return reject("depth distribution flattened", flattened_frames=len(flattened_frames),
                          checked_frames=checked_frames, **worst)

        # Fixing only one pose does not constrain the monocular scale gauge.
        if len(before_poses) >= 2 and checked_frames:
            anchor = self.select_fixed_frame_id(frames, 0)
            old_centers = {key: -pose[:3, :3].T @ pose[:3, 3] for key, pose in before_poses.items()}
            new_centers = {key: -pose[:3, :3].T @ pose[:3, 3] for key, pose in after_poses.items()}
            baselines = np.array([np.linalg.norm(old_centers[key] - old_centers[anchor]) for key in before_poses if key != anchor])
            new_baselines = np.array([np.linalg.norm(new_centers[key] - new_centers[anchor]) for key in before_poses if key != anchor])
            significant = baselines > max(float(baselines.max()) * 0.05, 1e-12)
            if np.any(significant):
                baseline_ratio = float(np.median(new_baselines[significant] / baselines[significant]))
                if baseline_ratio < self.settings.guard_baseline_ratio:
                    return reject("camera baseline collapse", baseline_ratio=baseline_ratio)
        return {"accepted": True, "reason": "accepted", "checked_frames": checked_frames,
                "min_depth_ratio": min_depth_ratio}


    #     # Perform bundle adjustment

    def bundle_adjustment(self, frames, merged_mappoints_dict, set_fix_id = 0, loop_kf = 0):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        focal_length = self.focal_length
        principal_point = (self.cx, self.cy)

        cam = g2o.CameraParameters(focal_length, principal_point, 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        maximum_frame_id = 0
        fixed_frame_id = self.select_fixed_frame_id(frames, set_fix_id)
        # Historical points can have only one observer inside this BA window.
        # Keep their existing geometry instead of fitting a free single-view point.
        fixed_history_points = set()
        for point_id, point in merged_mappoints_dict.items():
            observations = point.get_observations()
            observers = [
                key for key in observations
                if key in frames and frames[key].get_KeysUn_dict_feature(point_id) is not None
            ]
            # Historical points may have only one observation in the active
            # window. Their omitted observers cannot constrain a free vertex.
            if len(observers) < 2 and getattr(point, "local_recovery_protected", False):
                fixed_history_points.add(point_id)

        for frame_id, keyframe in frames.items():
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(frame_id)
            pose = g2o.Isometry3d(keyframe.get_pose())

            v_se3.set_estimate(g2o.SE3Quat(pose.rotation(), pose.translation()))


            if frame_id == fixed_frame_id:
                v_se3.set_fixed(True)
            else:
                v_se3.set_fixed(False)
            
            optimizer.add_vertex(v_se3)

            maximum_frame_id = max(maximum_frame_id, frame_id)

        # Prepare data structures for edges
        edges_mono = []
        index_edges_mono = []

        point_vertices = []

        point_id_dict = {}

        delta_mono = np.sqrt(self.settings.mono_chi2)

        point_vertex_ids = self.allocate_vertex_ids(
            frames.keys(), merged_mappoints_dict.keys()
        )
        for map_point_id, map_point in merged_mappoints_dict.items():
            point_id = point_vertex_ids[map_point_id]

            map_point_pos = map_point.get_world_pos()

            map_point_obs = map_point.get_observations().keys()
            point_vertex = g2o.VertexPointXYZ()
            point_vertex.set_id(point_id)  # Vertex IDs must be unique
            point_vertex.set_estimate(map_point_pos.reshape(-1))  # Initial estimate of the 3D point
            point_vertex.set_marginalized(True)  # Allow marginalization
            point_vertex.set_fixed(
                map_point_id in fixed_history_points
            )
            optimizer.add_vertex(point_vertex)
            point_vertices.append(point_vertex)

            ## example structure for the 2d feature dict

            for feats_with_frame_id in map_point_obs:
                if feats_with_frame_id in frames:
                    kf_frame = frames[feats_with_frame_id]
                    mv_keys_un = kf_frame.get_KeysUn_dict_feature(map_point_id)
                    if mv_keys_un is None:
                        continue
                    if mv_keys_un[0] < 0 or mv_keys_un[0] > self.width or mv_keys_un[1] < 0 or mv_keys_un[1] > self.height:
                        continue
                else:
                    continue

                obs = np.array([mv_keys_un[0], mv_keys_un[1]])
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, point_vertex)
                edge.set_vertex(1, optimizer.vertex(feats_with_frame_id))
                edge.set_measurement(obs)
                inv_sigma2 = 1.0
                edge.set_information(np.identity(2) * inv_sigma2)

                edge.set_parameter_id(0, 0)


                robust_kernel = g2o.RobustKernelHuber()
                edge.set_robust_kernel(robust_kernel)
                robust_kernel.set_delta(delta_mono)

            
                optimizer.add_edge(edge)

                edges_mono.append(edge)
                index_edges_mono.append(point_id)
                point_id_dict[map_point_id] = point_id

        optimizer.initialize_optimization()
        optimizer.optimize(self.settings.iterations)


        output_kfs = {}
        # KeyFrames
        for frame_id in frames.keys():
            vertex_se3 = optimizer.vertex(frame_id)
            pose = vertex_se3.estimate()
            cam_pose = self.isometry3d_to_numpy(pose)
            output_kfs[frame_id] = cam_pose

        output_mappoints = {}
        # MapPoints
        for map_point_id_key, point_id_read in point_id_dict.items():
            vertex_point = optimizer.vertex(point_id_read)
            if vertex_point is not None:
                pos = vertex_point.estimate()
                # g2o exposes borrowed Eigen storage without a Python owner.
                # Materialize it before the optimizer/vertex can be destroyed.
                optimized_map_pose = np.array(pos, dtype=float, copy=True)
                output_mappoints[map_point_id_key] = optimized_map_pose

        if not all(np.all(np.isfinite(value)) for value in list(output_kfs.values()) + list(output_mappoints.values())):
            _logger.warning('%s', "local BA rejected: non-finite solution; retaining input geometry")
            return (
                {key: frame.get_pose() for key, frame in frames.items()},
                {key: point.get_world_pos().reshape(3) for key, point in merged_mappoints_dict.items()},
            )
        return output_kfs, output_mappoints


    def optimize_essential_graph_3(self, allKFs, correctedloop, close_loop_candidates):
        # Setup optimizer
        optimizer = g2o.SparseOptimizer()
        optimizer.set_verbose(False)

        solver = g2o.OptimizationAlgorithmLevenberg(
            g2o.BlockSolverX(
                g2o.LinearSolverEigenX()
            )
        )

        optimizer.set_algorithm(solver)

        nMaxKFid = len(allKFs)

        vScw = {}
        vCorrectedSwc = {}
        vpVertices = {}

        minFeat = 100


        for ids, pKF in allKFs.items():
            VSim3 = g2o.VertexSim3Expmap()
            
            nIDi = ids

            Tcw = pKF

            Rcw = Tcw[:3,:3]
            tcw = Tcw[:3, 3].reshape(-1)
            Siw = g2o.Sim3(Rcw, tcw, 1.0)
            vScw[nIDi] = Siw
            VSim3.set_estimate(Siw)


            if nIDi in close_loop_candidates:
                VSim3.set_fixed(True)

            VSim3.set_id(nIDi)
            VSim3.set_marginalized(False)


            optimizer.add_vertex(VSim3)
            vpVertices[nIDi] = VSim3

        sInsertedEdges = set()
        matLambda = np.identity(7)

        # set normal edges
        for pKF in correctedloop.values():
            nIDi = pKF["ref"]
            nIDj = pKF["cur"]
            Rcw = pKF["Rci"]
            tcw = pKF["tci"]

            Sji = g2o.Sim3(Rcw, tcw, 1.0)

            vertex_i = optimizer.vertex(nIDi)
            vertex_j = optimizer.vertex(nIDj)
            if vertex_i is None or vertex_j is None:
                continue

            e = g2o.EdgeSim3()
            e.set_vertex(1, vertex_j)
            e.set_vertex(0, vertex_i)
            e.set_measurement(Sji)
            e.set_information(matLambda)

            optimizer.add_edge(e)
            sInsertedEdges.add((min(nIDi, nIDj), max(nIDi, nIDj)))

        
        # Optimize
        optimizer.initialize_optimization()
        optimizer.optimize(self.settings.iterations)

        # SE3 Pose Recovery
        new_KFs = {}
        for ids in allKFs.keys():
            nIDi = ids
            VSim3 = optimizer.vertex(nIDi)
            CorrectedSiw = VSim3.estimate()
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse()
            eigR = CorrectedSiw.rotation().matrix()
            eigt = CorrectedSiw.translation()
            s = CorrectedSiw.scale()
            eigt /= s
            

            Tiw = np.identity(4)
            Tiw[:3,:3] = eigR
            Tiw[:3, 3] = eigt.reshape(-1)

            new_KFs[ids] = Tiw

        return new_KFs


    def local_bundle_adjustment_dense_2(self, frames_pose:dict, mappoints:dict, ref_id, target_id):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        principal_point = (self.cx, self.cy)

        cam = g2o.CameraParameters(self.focal_length, principal_point, 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        n_initial_correspondences = 0

        for frame_id, frame_pose in frames_pose.items():
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(frame_id)
            pose = g2o.Isometry3d(frame_pose)

            
            if frame_id == self.init_frame_id:
                v_se3.set_fixed(True)
                pose = g2o.Isometry3d(np.identity(4))
                v_se3.set_estimate(g2o.SE3Quat(pose.rotation(), pose.translation()))
            else:
                v_se3.set_fixed(False)
                v_se3.set_estimate(g2o.SE3Quat(pose.rotation(), pose.translation()))

            optimizer.add_vertex(v_se3)

        # Prepare data structures for edges
        edges_mono = []
        index_edges_mono = []

        point_vertices = []

        point_id_dict = {}

        delta_mono = np.sqrt(self.settings.mono_chi2)

        point_vertex_ids = self.allocate_vertex_ids(
            frames_pose.keys(), mappoints.keys()
        )
        for map_point_id, point in mappoints.items():
            point_id = point_vertex_ids[map_point_id]
            map_point = point["map_point"]
            feats = point["feats"]
            point_vertex = g2o.VertexPointXYZ()
            point_vertex.set_id(point_id)  # Vertex IDs must be unique
            point_vertex.set_estimate(map_point.reshape(-1))  # Initial estimate of the 3D point
            point_vertex.set_marginalized(True)  # Allow marginalization
            optimizer.add_vertex(point_vertex)
            point_vertices.append(point_vertex)

            ## example structure for the 2d feature dict

            for feats_with_frame_id in feats:
                mv_keys_un = list(feats_with_frame_id.values())[0]
                feats_frame_id = list(feats_with_frame_id.keys())[0]
                if mv_keys_un[0] < 0 or mv_keys_un[0] > 720 or mv_keys_un[1] < 0 or mv_keys_un[1] > 720:
                    continue
                frame_vertex = optimizer.vertex(feats_frame_id)
                if frame_vertex is None:
                    continue
                obs = np.array([mv_keys_un[0], mv_keys_un[1]])
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, point_vertex)
                edge.set_vertex(1, frame_vertex)
                edge.set_measurement(obs)
                inv_sigma2 = 1.0
                edge.set_information(np.identity(2) * inv_sigma2)

                edge.set_parameter_id(0, 0)


                robust_kernel = g2o.RobustKernelHuber()
                edge.set_robust_kernel(robust_kernel)
                robust_kernel.set_delta(delta_mono)

            
                optimizer.add_edge(edge)

                edges_mono.append(edge)
                index_edges_mono.append(point_id)
                point_id_dict[map_point_id] = point_id

        optimizer.initialize_optimization()
        optimizer.optimize(self.settings.iterations)


        output_kfs = {}
        # KeyFrames
        for frame_id in frames_pose.keys():
            vertex_se3 = optimizer.vertex(frame_id)
            pose = vertex_se3.estimate()
            cam_pose = self.isometry3d_to_numpy(pose)
            output_kfs[frame_id] = cam_pose

        output_mappoints = {}
        # MapPoints
        for map_point_id_key, point_id_read in point_id_dict.items():
            vertex_point = optimizer.vertex(point_id_read)
            if vertex_point is not None:
                pos = vertex_point.estimate()
                optimized_map_pose = pos
                output_mappoints[map_point_id_key] = optimized_map_pose
        

        return output_kfs, output_mappoints
    

    def pose_optimization4(self, mvpMapPoints:dict, undist_features:dict, pose = np.identity(4)):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        principal_point = (self.cx, self.cy)

        cam = g2o.CameraParameters(self.focal_length, principal_point, 0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        n_initial_correspondences = 0

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(0)
        pose = g2o.Isometry3d(pose)


        v_se3.set_estimate(g2o.SE3Quat(pose.rotation(), pose.translation()))
        v_se3.set_fixed(False)
        optimizer.add_vertex(v_se3)

        # Prepare data structures for edges
        edges_mono = []
        index_edges_mono = []
        edges_stereo = []
        index_edges_stereo = []

        delta_mono = np.sqrt(self.settings.mono_chi2)
        delta_stereo = np.sqrt(self.settings.stereo_chi2)

        point_vertices = []
        point_vertex_ids = self.allocate_vertex_ids([0], mvpMapPoints.keys())
        for key, value in mvpMapPoints.items():
            map_point = value
            if key in undist_features:
                mv_keys_un = undist_features[key].copy()
                if mv_keys_un[0] < 0 or mv_keys_un[0] > 720 or mv_keys_un[1] < 0 or mv_keys_un[1] > 720:
                    continue
                if map_point is not None:
                    n_initial_correspondences += 1

                    point_vertex = g2o.VertexPointXYZ()
                    point_vertex.set_id(point_vertex_ids[key])
                    point_vertex.set_estimate(map_point.reshape(-1))  # Initial estimate of the 3D point
                    point_vertex.set_marginalized(True)  # Allow marginalization
                    # This is pose estimation against an existing map. Free
                    # single-view points make both pose and scale unobservable.
                    point_vertex.set_fixed(True)
                    optimizer.add_vertex(point_vertex)
                    point_vertices.append(point_vertex)

                    
                    obs = np.array([mv_keys_un[0], mv_keys_un[1]])
                    edge = g2o.EdgeProjectXYZ2UV()
                    edge.set_vertex(0, point_vertex)
                    edge.set_vertex(1, v_se3)
                    edge.set_measurement(obs)
                    inv_sigma2 = 1.0
                    edge.set_information(np.identity(2) * inv_sigma2)

                    edge.set_parameter_id(0, 0)


                    robust_kernel = g2o.RobustKernelHuber()
                    edge.set_robust_kernel(robust_kernel)
                    robust_kernel.set_delta(delta_mono)

                    
                    optimizer.add_edge(edge)
                    edges_mono.append(edge)
                    index_edges_mono.append(point_vertex_ids[key])
        
        optimizer.initialize_optimization()
        optimizer.optimize(self.settings.iterations)

        optimized_pose = v_se3.estimate()
        Rcw = optimized_pose.rotation().matrix()
        tcw = optimized_pose.translation()
        
        return n_initial_correspondences, Rcw, tcw
