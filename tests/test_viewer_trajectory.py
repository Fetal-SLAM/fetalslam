import numpy as np

from fetal_slam.viewer import PangolinViewer


def test_initial_trajectory_uses_normalized_camera_poses():
    viewer = PangolinViewer.__new__(PangolinViewer)
    viewer.trajectory = [np.array([999.0, 0.0, 0.0])]
    first_tcw = np.eye(4)
    second_tcw = np.eye(4)
    second_tcw[0, 3] = 1.0

    viewer.set_initial_trajectory((first_tcw, second_tcw))

    assert len(viewer.trajectory) == 2
    assert np.allclose(viewer.trajectory[0], [0.0, 0.0, 0.0])
    assert np.allclose(viewer.trajectory[1], [-1.0, 0.0, 0.0])
    assert all(np.linalg.norm(point) < 2.0 for point in viewer.trajectory)
