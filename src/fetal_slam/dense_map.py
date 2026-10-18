import numpy as np
import threading
import time
import pickle
import open3d as o3d
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay, cKDTree
import cv2
from scipy.spatial.transform import Rotation
import torch
from fetal_slam.dense_config import dense_settings
from fetal_slam.plane_mesh import build_plane_mesh

import logging as _logging

_logger = _logging.getLogger(__name__)

class DenseMap:
    def __init__(self):

        self.settings = dense_settings()
        self.stop_patience = self.settings['stop_patience']
        self.residual_tolerance = self.settings['residual_tolerance']

        self.intrinsic_matrix = None

        self.margin = self.settings['margin']

        self.width = None
        self.height = None
        self.min_views = self.settings['min_views']
        self.device = self.settings['device']
        self.cancel_event = None

    def check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise InterruptedError("Dense map cancelled")

    def run_snapshot(self, snapshot, cancel_event, progress=lambda value: None):
        """Headless surface interpolation and pose-only photometric refinement."""
        self.cancel_event = cancel_event
        settings = dense_settings({key: snapshot[key] for key in self.settings if key in snapshot})
        self.settings = settings
        self.stop_patience = settings['stop_patience']
        self.residual_tolerance = settings['residual_tolerance']
        self.intrinsic_matrix = np.asarray(snapshot["intrinsic"], dtype=float).copy()
        if (self.intrinsic_matrix.shape != (3, 3) or
                not np.isfinite(self.intrinsic_matrix).all() or
                min(self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]) <= 0):
            raise ValueError('Invalid calibrated intrinsic matrix')
        self.margin = settings['margin']
        self.min_views = settings['min_views']
        self.device = settings['device']
        self.cam_traj_dict_image = snapshot["keyframes"]
        if len(self.cam_traj_dict_image) < 2:
            raise ValueError("Dense map needs at least two keyframes")
        first = next(iter(self.cam_traj_dict_image.values()))
        self.height, self.width = first["image"].shape[:2]
        for frame in self.cam_traj_dict_image.values():
            pose = np.asarray(frame['pose'])
            if frame['image'].shape[:2] != (self.height, self.width):
                raise ValueError('Keyframe image sizes must match the calibrated camera')
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError('Invalid keyframe Tcw extrinsic')
        self.map_points = np.asarray(snapshot["points"], dtype=float).reshape(-1, 3).copy()
        self.map_points = np.unique(self.map_points[np.isfinite(self.map_points).all(axis=1)], axis=0)
        if len(self.map_points) < settings['min_sparse_points']:
            raise ValueError('Too few finite sparse points for dense reconstruction')
        self.check_cancelled()
        progress("Surface mesh")

        if settings['surface_mode'] == 'plane_grid':
            mesh, report = build_plane_mesh(self.map_points, settings, self.check_cancelled)
            _logger.info('%s', f"Plane mesh: {report['inliers']}/{report['input_points']} inliers, "
                  f"grid={report['grid_shape']}, supported={report['supported_vertices']}")
        else:
            centered = self.map_points - np.median(self.map_points, axis=0)
            _, singular, axes = np.linalg.svd(centered, full_matrices=False)
            if singular[1] < max(singular[0] * settings['collinear_ratio'], 1e-12):
                raise ValueError("Sparse points are collinear")
            chart, keep = np.unique(centered @ axes[:2].T, axis=0, return_index=True)
            vertices = self.map_points[keep]
            triangles = Delaunay(chart).simplices
            spacing = np.median(cKDTree(chart).query(chart, k=2)[0][:, 1])
            edges = chart[triangles] - np.roll(chart[triangles], 1, axis=1)
            triangles = triangles[np.max(np.linalg.norm(edges, axis=2), axis=1) <= settings['mesh_edge_factor'] * spacing]
            if not len(triangles):
                raise ValueError("No supported surface triangles")
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
            )
        self.num_points = settings['num_points']
        progress("Dense sampling")
        self.check_cancelled()
        cloud = self.dense_map_from_mesh(mesh)
        if len(cloud) < settings['min_dense_points']:
            raise ValueError("Too few visible dense points")
        progress("PBA")
        poses = self.pba_mesh(cloud, optimization_itr_num=settings['iterations'])
        self.check_cancelled()
        progress("Color fusion")
        points_h = np.column_stack((cloud[:, :3], np.ones(len(cloud))))
        colors, visible = self.update_point_cloud_rgb(
            self.cam_traj_dict_image, poses, points_h, self.intrinsic_matrix, cloud[:, 3:]
        )
        visible &= np.isfinite(colors).all(axis=1)
        if not np.any(visible):
            raise ValueError("No visible points after PBA")
        return {"points": cloud[visible, :3].copy(),
                "colors": np.clip(colors[visible, ::-1] / 255.0, 0, 1),
                "poses": poses, "pba": self.pba_report}

    def load_cam_traj(self, cam_traj_save_file):
        with open(cam_traj_save_file, 'rb') as file:
            self.cam_traj_dict_image = pickle.load(file)

    def load_map_points(self, map_points_save_file):
        points = np.load(map_points_save_file)

        if not np.isfinite(points).all():
            raise ValueError("Points contain NaNs or Infs")

        self.map_points = points.reshape(-1, 3)

    def dense_map_from_mesh(self, mesh):
        pcd = mesh.sample_points_uniformly(number_of_points=self.num_points)

        pcd_points = np.asarray(pcd.points)

        pcd_colors = np.zeros_like(pcd_points)
        visibility = np.zeros(len(pcd_points), dtype=int)

        cam_id_list = list(self.cam_traj_dict_image.keys())

        for cam_id in cam_id_list:
            self.check_cancelled()

            image_rgb = self.cam_traj_dict_image[cam_id]["image"].copy()
            image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
            cam_pose_moving = self.cam_traj_dict_image[cam_id]["pose"]

            map_points_arr = np.asarray(pcd.points).copy()

            points_h = np.hstack((map_points_arr, np.ones((map_points_arr.shape[0], 1))))

            points_camera_frame = (cam_pose_moving @ points_h.T).T

            image_points = self.intrinsic_matrix @ points_camera_frame[:, :3].T
            image_points /= image_points[2]
            image_coords = image_points[:2].T

            height, width, _ = image.shape

            valid_indices = np.where((image_coords[:, 0] >= self.margin) & (image_coords[:, 0] < width-self.margin) &
                                    (image_coords[:, 1] >= self.margin) & (image_coords[:, 1] < height-self.margin))[0]

            valid_indices = np.intersect1d(valid_indices, np.where(points_camera_frame[:, 2] > self.settings['sampling_min_depth'])[0])

            image_coords_int = np.clip(np.round(image_coords[valid_indices]).astype(int), [0, 0], [width - 1, height - 1])

            point_colors = image[image_coords_int[:, 1], image_coords_int[:, 0]] / 255.0

            first_seen = visibility[valid_indices] == 0
            pcd_colors[valid_indices[first_seen]] = point_colors[first_seen]
            visibility[valid_indices] += 1

        self.width = width
        self.height = height

        all_dense_map = np.hstack((pcd_points, pcd_colors))

        return all_dense_map[visibility >= self.min_views]

    def se3_to_T(self, delta_se3):
        """
        Convert a 6D SE(3) vector (Lie Algebra) to a 4x4 transformation matrix.

        Args:
            delta_se3 (numpy.ndarray): 6D vector [ωx, ωy, ωz, tx, ty, tz]

        Returns:
            numpy.ndarray: 4x4 transformation matrix in SE(3)
        """
        delta_se3 = np.asarray(delta_se3).flatten()

        omega = delta_se3[:3]
        translation = delta_se3[3:]

        theta = np.linalg.norm(omega)
        if theta < 1e-8:
            R_mat = np.eye(3)
        else:
            R_mat = Rotation.from_rotvec(omega).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = translation

        return T

    def get_pixel_values_batch(self, image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Fetch grayscale values at multiple (u, v) positions using bilinear interpolation in PyTorch.

        Parameters:
            image: 2D PyTorch tensor of shape (H, W) representing the grayscale image.
            u: 1D PyTorch tensor of shape (N,) containing x-coordinates.
            v: 1D PyTorch tensor of shape (N,) containing y-coordinates.

        Returns:
            values: 1D PyTorch tensor of shape (N,) with interpolated intensity values.
        """

        u = u.float()
        v = v.float()

        height, width = image.shape

        u_floor, u_ceil = u.floor().long(), torch.clamp(u.ceil().long(), max=width - 1)
        v_floor, v_ceil = v.floor().long(), torch.clamp(v.ceil().long(), max=height - 1)

        u_floor = torch.clamp(u_floor, 0, width - 1)
        v_floor = torch.clamp(v_floor, 0, height - 1)

        I_ff = image[v_floor, u_floor]
        I_fc = image[v_floor, u_ceil]
        I_cf = image[v_ceil, u_floor]
        I_cc = image[v_ceil, u_ceil]

        u_ratio = u - u_floor.float()
        v_ratio = v - v_floor.float()

        values = (I_ff * (1 - u_ratio) * (1 - v_ratio) +
                I_fc * u_ratio * (1 - v_ratio) +
                I_cf * (1 - u_ratio) * v_ratio +
                I_cc * u_ratio * v_ratio)

        return values

    def optimize_camera_trajectory_point_matrix(self, opt_camera_imgs:dict, opt_camera_traj:dict, intrinsic, points, intensities, fixed_pose_id, optimization_itr_num, device):
        total_start_time = time.time()
        optimization_time_list = []

        intrinsic_torch = intrinsic

        intensities_torch = intensities

        points_h_torch = points

        prev_residual_norm = float('inf')
        stable_iterations = 0

        self.pba_report = {"accepted_steps": 0, "residuals": [], "fixed_pose_id": fixed_pose_id,
                           "iterations": 0, "stop_reason": "max_iterations"}

        for i in range(optimization_itr_num):
            self.check_cancelled()
            _logger.debug('%s', f"optimize iter {i}.")
            iter_start_time = time.time()


            total_residuals = []

            for cam_id in opt_camera_imgs.keys():
                self.check_cancelled()
                if cam_id == fixed_pose_id:
                    continue

                pose_torch = torch.tensor(opt_camera_traj[cam_id]["pose"], dtype=torch.float32, device=device)
                img_gray = opt_camera_imgs[cam_id]["gray_image"]
                grad_x = opt_camera_imgs[cam_id]["grad_x"]
                grad_y = opt_camera_imgs[cam_id]["grad_y"]

                g = torch.matmul(pose_torch, points_h_torch.T).T

                uv = torch.matmul(intrinsic_torch, g[:, :3].T).T
                u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]

                valid_mask = (u >= self.margin) & (u < self.width - self.margin) & (v >= self.margin) & (v < self.height - self.margin)
                valid_mask &= (g[:, 2] > self.settings['pba_min_depth']) & torch.isfinite(g).all(dim=1)
                if int(valid_mask.sum()) < self.settings['min_pba_observations']:
                    continue

                u, v, g = u[valid_mask], v[valid_mask], g[valid_mask]


                gray_values = self.get_pixel_values_batch(img_gray, u, v)
                grad_x_values = self.get_pixel_values_batch(grad_x, u, v)
                grad_y_values = self.get_pixel_values_batch(grad_y, u, v)

                residuals = gray_values - intensities_torch[valid_mask]
                total_residuals.append(residuals)

                invz = 1.0 / g[:, 2]
                v0 = grad_x_values * intrinsic_torch[0, 0] * invz
                v1 = grad_y_values * intrinsic_torch[1, 1] * invz
                v2 = -(v0 * g[:, 0] + v1 * g[:, 1]) * invz

                J_r = torch.zeros((len(u), 6), dtype=torch.float32, device=device)
                J_r[:, 0] = -g[:, 2] * v1 + g[:, 1] * v2
                J_r[:, 1] = g[:, 2] * v0 - g[:, 0] * v2
                J_r[:, 2] = -g[:, 1] * v0 + g[:, 0] * v1
                J_r[:, 3] = v0
                J_r[:, 4] = v1
                J_r[:, 5] = v2

                JTJ = torch.matmul(J_r.T, J_r)
                JTr = torch.matmul(J_r.T, residuals.view(-1, 1))

                try:
                    damping = self.settings['pba_damping'] * torch.clamp(torch.diagonal(JTJ).mean(), min=1.0)
                    delta = torch.linalg.solve(JTJ + damping * torch.eye(6, device=device), JTr).squeeze()
                    if not torch.isfinite(delta).all():
                        continue

                    delta_pose = delta[:6]
                    rotation_norm = torch.linalg.norm(delta_pose[:3])
                    translation_norm = torch.linalg.norm(delta_pose[3:])
                    bound = self.settings['max_translation_depth_ratio'] * torch.median(g[:, 2])
                    scale = min(1.0, self.settings['max_rotation_step'] / max(float(rotation_norm), 1e-12),
                                float(bound) / max(float(translation_norm), 1e-12))
                    delta_pose = delta_pose * scale

                    for attempt in range(self.settings['pba_backtracking_steps']):
                        self.check_cancelled()
                        delta_matrix = self.se3_to_T((delta_pose * (0.5 ** attempt)).cpu().numpy())
                        update_pose = np.linalg.inv(delta_matrix) @ opt_camera_traj[cam_id]["pose"]
                        trial = torch.tensor(update_pose, dtype=torch.float32, device=device) @ points_h_torch[valid_mask].T
                        if not torch.isfinite(trial).all() or not bool((trial[2] > self.settings['pba_min_depth']).all()):
                            continue
                        projected = intrinsic_torch @ trial[:3]
                        trial_u, trial_v = projected[0] / projected[2], projected[1] / projected[2]
                        inside = ((trial_u >= self.margin) & (trial_u < self.width - self.margin) &
                                  (trial_v >= self.margin) & (trial_v < self.height - self.margin))
                        if not self.projection_overlap_valid(inside):
                            continue
                        # Compare exactly the same points before/after the trial.
                        trial_residual = (self.get_pixel_values_batch(img_gray, trial_u[inside], trial_v[inside])
                                          - intensities_torch[valid_mask][inside])
                        if not torch.isfinite(trial_residual).all() or torch.mean(trial_residual ** 2) > torch.mean(residuals[inside] ** 2):
                            continue
                        opt_camera_traj[cam_id]["pose"] = update_pose
                        self.pba_report["accepted_steps"] += 1
                        break

                except RuntimeError:
                    _logger.debug('%s', "Singular matrix encountered.")

            if not total_residuals:
                raise ValueError("PBA has no valid photometric observations")
            total_residuals = torch.cat(total_residuals)
            residual_norm = torch.mean(total_residuals**2).item()
            self.pba_report["residuals"].append(residual_norm)
            self.pba_report['iterations'] = i + 1
            if not np.isfinite(residual_norm):
                raise ValueError('PBA residual is non-finite')

            _logger.debug('%s', f"Residual norm: {residual_norm:.6f}")

            stable_iterations = self.count_stable_iterations(
                prev_residual_norm, residual_norm, stable_iterations)
            if stable_iterations >= self.stop_patience:
                self.pba_report['stop_reason'] = 'residual_plateau'
                optimization_time_list.append(time.time() - iter_start_time)
                _logger.info('%s', f'Stopping PBA after {i + 1} iterations: residual change <= '
                      f'{self.residual_tolerance} for {stable_iterations} consecutive iterations.')
                break

            prev_residual_norm = residual_norm

            iter_time = time.time() - iter_start_time
            optimization_time_list.append(iter_time)
            _logger.debug('%s', f"Iteration {i+1}/{optimization_itr_num} completed in {iter_time:.4f} seconds.")


        total_time = time.time() - total_start_time
        _logger.info('%s', f"Total computation time: {total_time:.4f} seconds.")
        mean_optimization_time = np.array(optimization_time_list).mean()
        _logger.debug('%s', f"Mean computation time: {mean_optimization_time:.8f} seconds.")

        return opt_camera_traj

    def projection_overlap_valid(self, inside):
        count = int(inside.sum())
        return (count >= self.settings['min_pba_observations'] and
                count >= self.settings['pba_min_overlap'] * inside.numel())

    def count_stable_iterations(self, previous, current, count):
        if np.isfinite(previous) and abs(previous - current) <= self.residual_tolerance:
            return count + 1
        return 0

    def pba_mesh(self, color_point_cloud, optimization_itr_num=None):
        if optimization_itr_num is None:
            optimization_itr_num = self.settings['iterations']
        if not 1 <= optimization_itr_num <= 500:
            raise ValueError('PBA iteration limit must be between 1 and 500')
        device = self.device
        if device == 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _logger.debug('%s', f"Using device: {device}")

        K = torch.tensor(self.intrinsic_matrix, dtype=torch.float32, device=device)

        points = torch.tensor(color_point_cloud[:, :3], dtype=torch.float32, device=device)
        colors_rgb = torch.tensor(color_point_cloud[:, 3:], dtype=torch.float32, device=device) * 255

        weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=device)
        intensities = torch.matmul(colors_rgb, weights)

        ones = torch.ones((points.shape[0], 1), dtype=torch.float32, device=device)
        points_h = torch.cat((points, ones), dim=1)

        cam_imgs_for_opt = {}
        cam_traj = {}

        cam_id_list = list(self.cam_traj_dict_image.keys())

        fixed_pose_id = cam_id_list[0]

        for counter, cam_id in enumerate(cam_id_list):

            _logger.debug('%s', f"Processing camera {counter}: ID {cam_id}")

            image = self.cam_traj_dict_image[cam_id]["image"].copy()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.settings['gradient_kernel'], scale=self.settings['gradient_scale'])
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.settings['gradient_kernel'], scale=self.settings['gradient_scale'])

            gray_image_torch = torch.tensor(gray_image, dtype=torch.float32, device=device)
            grad_x_torch = torch.tensor(grad_x, dtype=torch.float32, device=device)
            grad_y_torch = torch.tensor(grad_y, dtype=torch.float32, device=device)

            cam_pose_moving = self.cam_traj_dict_image[cam_id]["pose"]

            cam_imgs_for_opt[cam_id] = {
                "gray_image": gray_image_torch,
                "grad_x": grad_x_torch,
                "grad_y": grad_y_torch

            }
            cam_traj[cam_id] = {
                "pose": cam_pose_moving
            }

        optimized_poses = cam_traj.copy()
        optimized_point_intensity = intensities.clone()

        optimized_poses = self.optimize_camera_trajectory_point_matrix(
            cam_imgs_for_opt, optimized_poses, K, points_h, optimized_point_intensity, fixed_pose_id, optimization_itr_num, device=device
        )

        optimized_poses_dict = {cam_id: {"pose": optimized_poses[cam_id]["pose"]} for cam_id in optimized_poses.keys()}

        return optimized_poses_dict

    def update_point_cloud_rgb(self, cam_img_traj, opt_camera_traj, points_h, intrinsic, colors):
        """
        Update the RGB color of point clouds based on the new camera poses.
        """
        new_proxy_colors = np.ones_like(colors, dtype=np.float64)
        color_sum = np.zeros_like(colors, dtype=np.float64)
        color_count = np.zeros(len(colors), dtype=int)
        optimization_mask = np.zeros(colors.shape[0], dtype=bool)

        width = self.width
        height = self.height

        for cam_id in opt_camera_traj:
            self.check_cancelled()

            pose = opt_camera_traj[cam_id]["pose"]
            image_rgb = cam_img_traj[cam_id]["image"]

            g = (pose @ points_h.T).T

            uv = (intrinsic @ g[:, :3].T).T
            u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]

            valid_mask = np.where((u >= self.margin) & (u < width-self.margin) & (v >= self.margin) & (v < height-self.margin) & (g[:, 2] > self.settings['fusion_min_depth']) & np.isfinite(g).all(axis=1))
            u, v = u[valid_mask], v[valid_mask]
            optimization_mask[valid_mask] = True

            new_r = map_coordinates(image_rgb[:, :, 0], [v, u], order=1, mode='nearest')
            new_g = map_coordinates(image_rgb[:, :, 1], [v, u], order=1, mode='nearest')
            new_b = map_coordinates(image_rgb[:, :, 2], [v, u], order=1, mode='nearest')

            new_rgb = np.stack([new_r, new_g, new_b], axis=1)

            color_sum[valid_mask] += new_rgb
            color_count[valid_mask] += 1
            new_proxy_colors[valid_mask] = color_sum[valid_mask] / color_count[valid_mask][:, None]

        return new_proxy_colors, optimization_mask

if __name__ == '__main__':
    import argparse
    from fetal_slam.parameters import Parameter
    from fetal_slam.dense_job import show_dense_result

    parser = argparse.ArgumentParser(description='Reconstruct trusted saved SLAM keyframes with explicit calibration')
    parser.add_argument('data_folder')
    parser.add_argument('sys_param')
    parser.add_argument('cam_param')
    args = parser.parse_args()
    parameters = Parameter()
    parameters.load_sys_param(args.sys_param, args.cam_param)
    intrinsic, distortion = parameters.get_cam_K_distco()
    size = (parameters.cam_intrinsic.width, parameters.cam_intrinsic.height)
    calibrated, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, size, 0, size)
    mapper = DenseMap()
    mapper.load_cam_traj(f'{args.data_folder}/keyframe_traj.pkl')
    mapper.load_map_points(f'{args.data_folder}/map_points.npy')
    frames = dict(sorted(mapper.cam_traj_dict_image.items()))
    limit = parameters.dense_config['max_keyframes']
    if limit and len(frames) > limit:
        keys = list(frames)
        frames = {keys[i]: frames[keys[i]] for i in np.linspace(0, len(keys) - 1, limit, dtype=int)}
    result = mapper.run_snapshot(
        {**parameters.dense_config, 'intrinsic': calibrated,
         'keyframes': frames, 'points': mapper.map_points}, threading.Event(), print)
    show_dense_result(result, parameters.dense_config)

