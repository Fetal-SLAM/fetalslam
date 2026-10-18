import threading

import numpy as np

from .dense_map import DenseMap
from .dense_config import dense_settings

import logging as _logging

_logger = _logging.getLogger(__name__)


def snapshot_dense_scene(keyframes, map_points, intrinsic, max_keyframes=None, **settings):
    """Copy final Tcw poses and matching images after SLAM workers stop."""
    if max_keyframes is not None:
        settings['max_keyframes'] = max_keyframes
    settings = dense_settings(settings)
    max_keyframes = settings['max_keyframes']
    ordered = sorted(keyframes.values(), key=lambda frame: frame.mnFrameId)
    if max_keyframes and len(ordered) > max_keyframes:
        ordered = [ordered[index] for index in np.linspace(0, len(ordered) - 1, max_keyframes, dtype=int)]
    frames = {frame.mnId: {'pose': frame.get_pose().copy(), 'image': frame.image.copy()}
              for frame in ordered}
    points = [point.get_world_pos().reshape(3).copy() for point in map_points.values()
              if not point.is_bad() and point.Observations() >= settings['min_point_observations']]
    return {**settings, 'keyframes': frames, 'points': np.asarray(points).reshape(-1, 3),
            'intrinsic': np.asarray(intrinsic).copy()}


def show_dense_result(result, settings=None):
    import open3d as o3d
    settings = dense_settings(settings)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(result['points'])
    cloud.colors = o3d.utility.Vector3dVector(result['colors'])
    o3d.visualization.draw_geometries([cloud], window_name='Dense map + PBA',
                                    width=settings['viewer_width'], height=settings['viewer_height'])


def run_dense_after_sequence(snapshot_factory, settings):
    """Ask on the GUI thread; build only on consent, then open Open3D."""
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        _logger.warning('%s', 'DenseMap dialog unavailable: install python3-tk (system) or tk (Conda).')
        return None
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        accepted = messagebox.askyesno(
            'Dense map', 'Sequence finished. Reconstruct dense map and run PBA?\n'
            f"Target samples: {settings['num_points']:,}", parent=root)
        if not accepted:
            return None
        root.destroy()
        root = None
        result = DenseMap().run_snapshot(
            snapshot_factory(), threading.Event(), lambda stage: _logger.info('%s', f'DenseMap: {stage}'))
        _logger.info('%s', f"DenseMap complete: {len(result['points'])} points; PBA: {result['pba']}")
        show_dense_result(result, settings)
        return result
    except Exception as error:
        _logger.warning('%s', f'DenseMap failed: {error}')
        try:
            messagebox.showerror('Dense map failed', str(error), parent=root)
        except tk.TclError:
            pass
        return None
    finally:
        if root is not None:
            root.destroy()
