import numpy as np
import threading

import logging as _logging

_logger = _logging.getLogger(__name__)
class Map_point_index:
    mnNextID = 0
    def get_current_id():
        cur_id = Map_point_index.mnNextID
        Map_point_index.update_id()
        return cur_id
    def update_id():
        Map_point_index.mnNextID += 1

class MapPoint:
    def __init__(self, pos, ref_kf = None):
        """
        Initializes a MapPoint object.
        :param pos: Initial position (numpy array, shape (3, 1)).
        :param ref_kf: Reference KeyFrame object.
        :param map_obj: Associated Map object.
        :param frame: Frame object (optional).
        :param idx_f: Feature index in the frame (optional).
        """
        self.mWorldPos = pos
        self.mutex_pos = threading.RLock()
        self.mutex_features = threading.RLock()
        self.mbBad = False
        self.mpRefKF = ref_kf
        # observed in keyframes
        self.mObservations = {}
        self.mnobserved = 0
        self.nObs = 0
        self.mnId = Map_point_index.get_current_id()
        self.color = (0, 0, 0)
        if ref_kf is not None:
            self.Ow = ref_kf.get_camera_center()
            self.mNormalVector = self.mWorldPos - self.Ow
            self.mNormalVector /= np.linalg.norm(self.mNormalVector)

            pc = pos - self.Ow
            self.dist = np.linalg.norm(pc)
            self.level = 1
            self.level_scale_factor = 1.0


        else:
            _logger.debug('%s', "[Mappoint] current mappoint without a map thread!")

    def set_world_pos(self, pos):
        """
        Sets the world position of the MapPoint.
        :param pos: New position (numpy array, shape (3, 1)).
        """
        with self.mutex_pos:
            self.mWorldPos = pos.copy()

    def get_world_pos(self):
        """
        Returns the world position of the MapPoint.
        :return: World position (numpy array, shape (3, 1)).
        """
        with self.mutex_pos:
            return self.mWorldPos.copy()

    def get_map_point_id(self):
        return self.mnId

    def set_map_point_color(self, color):
        with self.mutex_features:
            self.color = color.copy()

    def get_map_point_color(self):
        with self.mutex_features:
            return self.color.copy()

    def get_normal(self):
        """
        Returns the normal vector of the MapPoint.
        :return: Normal vector (numpy array, shape (3, 1)).
        """
        return self.mNormalVector.copy()

    def get_reference_keyframe(self):
        """
        Returns the reference KeyFrame.
        :return: Reference KeyFrame object.
        """
        return self.mpRefKF

    def add_observation(self, keyframe_id):
        """
        Adds an observation of the MapPoint in a KeyFrame.
        :param keyframe_id: KeyFrame object.
        :param idx: Feature index in the KeyFrame.
        """
        with self.mutex_features:
            if keyframe_id in self.mObservations:
                return

            self.mObservations[keyframe_id] = True
            self.nObs += 1

    def erase_observation(self, keyframe_id):
        """
        Erases an observation from a KeyFrame.
        :param keyframe: KeyFrame object.
        """
        with self.mutex_features:
            if keyframe_id not in self.mObservations:
                return False
            del self.mObservations[keyframe_id]
            self.nObs = len(self.mObservations)

            if self.mpRefKF == keyframe_id:
                self.mpRefKF = next(iter(self.mObservations.keys()), None)
            return True


    def get_observations(self):
        """
        Returns the observations of the MapPoint.
        :return: Dictionary of KeyFrame objects to feature indices.
        """
        with self.mutex_features:
            return self.mObservations.copy()

    def Observations(self):
        return self.nObs

    def clear_all_observations(self):
        with self.mutex_features:
            self.mObservations = {}
            self.nObs = 0

    def set_bad_flag(self):
        obs = {}
        self.mbBad = True
        obs = self.mObservations.copy()
        self.mObservations.clear()

        for kf, idx in obs.items():
            kf.delete_map_point_match(idx)

    def is_bad(self):
        return self.mbBad

    def is_in_keyframe(self, KF):
        if KF in self.mObservations:
            return True
        else:
            return False
