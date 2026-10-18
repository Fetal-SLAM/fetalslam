import numpy as np
from .runtime_config import runtime_settings

try:
    import pangolin
    from OpenGL import GL as gl
except ImportError as error:
    raise ImportError(
        "The Pangolin viewer requires the 'pangolin' Python binding and PyOpenGL."
    ) from error


class PangolinViewer:
    """Interactive sparse-map viewer for the offline SLAM pipeline."""

    def __init__(self, width=None, height=None, window_name=None, settings=None):
        self.settings = settings or runtime_settings("viewer")
        width = self.settings.width if width is None else width
        height = self.settings.height if height is None else height
        window_name = self.settings.window_name if window_name is None else window_name
        self.width = width
        self.height = height
        self.window_name = window_name
        self.trajectory = []
        self.optimized_trajectory = []
        self.closed = False

        pangolin.CreateWindowAndBind(window_name, width, height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.96, 0.96, 0.96, 1.0)

        projection = pangolin.ProjectionMatrix(
            width, height, self.settings.focal_length, self.settings.focal_length, width / 2.0, height / 2.0, self.settings.near_clip, self.settings.far_clip
        )
        model_view = pangolin.ModelViewLookAt(
            0.0, -1.5, -3.0,
            0.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
        )
        self.camera = pangolin.OpenGlRenderState(projection, model_view)
        self.display = pangolin.CreateDisplay()
        self.display.SetBounds(
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            pangolin.Attach(0.0),
            pangolin.Attach(1.0),
            -float(width) / float(height),
        )
        self.display.SetHandler(pangolin.Handler3D(self.camera))

    def render(self, slam_map, current_tcw=None, keyframe_tree=None):
        if self.closed or pangolin.ShouldQuit():
            self.closed = True
            return False

        map_points, map_colors = self._map_point_snapshot(slam_map)
        keyframe_poses = self._keyframe_snapshot(slam_map, keyframe_tree)

        current_twc = None
        if current_tcw is not None:
            current_twc = np.linalg.inv(np.asarray(current_tcw, dtype=np.float64))
            self._append_trajectory_point(current_twc[:3, 3])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.display.Activate(self.camera)

        self._draw_axes()
        self._draw_map_points(map_points, map_colors)
        self._draw_trajectory(self.trajectory, (0.15, 0.15, 0.15), 2.0)
        self._draw_trajectory(self.optimized_trajectory, (0.9, 0.15, 0.1), 3.0)

        for pose in keyframe_poses:
            self._draw_camera(pose, (0.1, 0.35, 0.9), 0.08, 1.5)
        if current_twc is not None:
            self._draw_camera(current_twc, (0.1, 0.75, 0.2), 0.12, 3.0)

        pangolin.FinishFrame()
        return not pangolin.ShouldQuit()

    def set_optimized_trajectory(self, optimized_tcw_by_id):
        self.optimized_trajectory = []
        for frame_id in sorted(optimized_tcw_by_id):
            tcw = np.asarray(optimized_tcw_by_id[frame_id], dtype=np.float64)
            if tcw.shape == (4, 4):
                self.optimized_trajectory.append(np.linalg.inv(tcw)[:3, 3].copy())

    def set_initial_trajectory(self, initial_tcw_poses):
        """Start the live trajectory from normalized initialization poses."""
        self.trajectory = []
        for tcw in initial_tcw_poses:
            tcw = np.asarray(tcw, dtype=np.float64)
            if tcw.shape != (4, 4) or not np.all(np.isfinite(tcw)):
                continue
            self._append_trajectory_point(np.linalg.inv(tcw)[:3, 3])

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            pangolin.DestroyWindow(self.window_name)
        except RuntimeError:
            pass

    @staticmethod
    def _map_point_snapshot(slam_map):
        points = []
        colors = []
        for map_point in slam_map.get_all_map_points().values():
            if map_point is None:
                continue
            point = np.asarray(map_point.get_world_pos(), dtype=np.float64).reshape(-1)
            if point.size != 3 or not np.all(np.isfinite(point)):
                continue
            color = np.asarray(map_point.get_map_point_color(), dtype=np.float64).reshape(-1)
            if color.size != 3 or not np.all(np.isfinite(color)):
                color = np.array([40.0, 40.0, 40.0])
            points.append(point)
            colors.append(np.clip(color / 255.0, 0.0, 1.0))
        return points, colors

    @staticmethod
    def _keyframe_snapshot(slam_map, keyframe_tree):
        keyframes = {
            keyframe.get_keyframe_id(): keyframe
            for keyframe in slam_map.get_all_key_frames()
        }
        if keyframe_tree is not None:
            for keyframe in keyframe_tree.get_all_keyframes_for_loopclosing().values():
                keyframes[keyframe.get_keyframe_id()] = keyframe
            for keyframe in keyframe_tree.get_all_temp_keyframes():
                keyframes[keyframe.get_keyframe_id()] = keyframe
        return [keyframes[key].get_pose_inverse() for key in sorted(keyframes)]

    def _append_trajectory_point(self, point):
        point = np.asarray(point, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point)):
            return
        if not self.trajectory or np.linalg.norm(point - self.trajectory[-1]) > 1e-7:
            self.trajectory.append(point.copy())

    @staticmethod
    def _draw_axes():
        gl.glLineWidth(2.0)
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(0.85, 0.15, 0.15)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.5, 0.0, 0.0)
        gl.glColor3f(0.15, 0.7, 0.2)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.5, 0.0)
        gl.glColor3f(0.15, 0.3, 0.85)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.5)
        gl.glEnd()

    def _draw_map_points(self, points, colors):
        if not points:
            return
        gl.glPointSize(self.settings.point_size)
        gl.glBegin(gl.GL_POINTS)
        for point, color in zip(points, colors):
            gl.glColor3f(float(color[0]), float(color[1]), float(color[2]))
            gl.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
        gl.glEnd()

    @staticmethod
    def _draw_trajectory(points, color, width):
        if len(points) < 2:
            return
        gl.glLineWidth(width)
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_LINE_STRIP)
        for point in points:
            gl.glVertex3f(float(point[0]), float(point[1]), float(point[2]))
        gl.glEnd()

    @staticmethod
    def _draw_camera(twc, color, size, width):
        twc = np.asarray(twc, dtype=np.float64)
        if twc.shape != (4, 4) or not np.all(np.isfinite(twc)):
            return

        half_width = size
        half_height = size * 0.75
        depth = size * 0.6
        corners = [
            (-half_width, -half_height, depth),
            (half_width, -half_height, depth),
            (half_width, half_height, depth),
            (-half_width, half_height, depth),
        ]

        gl.glPushMatrix()
        gl.glMultMatrixd(twc.T)
        gl.glLineWidth(width)
        gl.glColor3f(*color)
        gl.glBegin(gl.GL_LINES)
        for corner in corners:
            gl.glVertex3f(0.0, 0.0, 0.0)
            gl.glVertex3f(*corner)
        for first, second in zip(corners, corners[1:] + corners[:1]):
            gl.glVertex3f(*first)
            gl.glVertex3f(*second)
        gl.glEnd()
        gl.glPopMatrix()
