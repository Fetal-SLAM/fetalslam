import cv2
import numpy as np

from fetal_slam.visual_reconstruction import VisualReconstruction
from fetal_slam.local_mapping import LocalMapping


def project(point, rotation, translation, intrinsic):
    camera_point = rotation @ point + translation
    pixel = intrinsic @ camera_point
    return np.array([[pixel[0, 0] / pixel[2, 0], pixel[1, 0] / pixel[2, 0]]])


def test_initial_triangulation_rejects_far_depth_outlier():
    intrinsic = np.array([[200.0, 0.0, 100.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]])
    rotation = np.eye(3)
    translation = np.array([[-1.0], [0.0], [0.0]])
    reconstruction = VisualReconstruction(
        intrinsic,
        width=200,
        height=200,
        min_point_parallax_degrees=0.1,
        max_depth_factor=5.0,
    )
    world_points = [np.array([[x], [0.0], [5.0]]) for x in np.linspace(-1.0, 1.0, 6)]
    world_points.append(np.array([[0.0], [0.0], [40.0]]))
    first = [project(point, np.eye(3), np.zeros((3, 1)), intrinsic) for point in world_points]
    second = [project(point, rotation, translation, intrinsic) for point in world_points]
    matches = [cv2.DMatch(index, index, 0) for index in range(len(world_points))]

    _, _, points, triangulated = reconstruction.CheckRT(
        rotation, translation, first, second, matches, intrinsic, th2=4.0
    )

    assert all(triangulated[:6])
    assert not triangulated[6]
    assert points[6] is None


def test_local_mapping_mask_clips_float_rectangles_to_image():
    mapping = LocalMapping(width=100, height=80)

    mask = mapping.create_mask([[-4.2, 10.7, 30.3, 90.8], [50.0, 20.0, 50.0, 30.0]])

    assert mask.shape == (80, 100)
    assert mask[11, 0] == 255
    assert mask[79, 30] == 255
    assert mask[25, 50] == 0
