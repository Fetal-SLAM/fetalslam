import threading

class Mapping:
    def __init__(self):
        """
        Initializes the Map object.
        """
        self.msp_map_points = {}  # Set of MapPoint objects
        self.msp_key_frames = set()  # Set of KeyFrame objects
        self.mvp_reference_map_points = {}  # List of reference MapPoints
        self.mn_max_kf_id = 0  # Max KeyFrame ID
        self.mn_big_change_idx = 0  # Index for significant changes in the map
        self.mutex_map = threading.RLock()

        self.mvp_key_frame_origins = []  # KeyFrame origins


    def add_key_frame(self, keyframe):
        """
        Adds a KeyFrame to the map.
        :param keyframe: KeyFrame object to add.
        """
        with self.mutex_map:
            self.msp_key_frames.add(keyframe)
            if keyframe.mnId > self.mn_max_kf_id:
                self.mn_max_kf_id = keyframe.mnId

    def add_map_point(self, map_point, idx):
        """
        Adds a MapPoint to the map.
        :param map_point: MapPoint object to add, idx of the mappoint.
        """
        with self.mutex_map:
            self.msp_map_points[idx] = map_point

    def add_map_points(self, map_points):
        """
        Adds a list of MapPoints to the map.
        :param map_points: List of MapPoint objects to add.
        """
        with self.mutex_map:
            for map_point in map_points:
                if map_point is not None:
                    self.msp_map_points[map_point.mnId] = map_point

    def update_map_point_world_pos(self, map_point_idx, map_point_pos):
        """
        Updates a MapPoint in the map.
        :param
        map_point_pos: New world position of the MapPoint.
        map_point_idx: Index of the MapPoint to update.
        """
        with self.mutex_map:
            map_point = self.msp_map_points.get(map_point_idx)
        if map_point is not None:
            map_point.set_world_pos(map_point_pos)

    def update_map_points_world_pos(self, map_points_pos:dict):
        '''
        Updates a list of MapPoints in the map.
        :param
        map_points_pos: New world position of the MapPoints.
        '''

        for idx, map_point_pos in map_points_pos.items():
            self.update_map_point_world_pos(idx, map_point_pos)


    def erase_map_point(self, mappoint_id):
        """
        Removes a MapPoint from the map.
        :param map_point: MapPoint object to remove.
        """
        with self.mutex_map:
            return self.msp_map_points.pop(mappoint_id, None)

    def erase_map_points(self, mappoint_ids):
        """
        Removes a MapPoint from the map.
        :param map_point: MapPoint object to remove.
        """
        with self.mutex_map:
            for mappoint_id in mappoint_ids:
                self.msp_map_points.pop(mappoint_id, None)

    def add_observation_in_map(self, keyframe_id, mp_id):
        '''
        Add an obeservation for a MapPoint in the map.
        '''
        self.msp_map_points[mp_id].add_observation(keyframe_id)

    def add_observations_in_map(self, keyframe_id, mp_ids):
        '''
        Add obeservations for a MapPoint in the map.
        '''
        for mp_id in mp_ids:
            self.msp_map_points[mp_id].add_observation(keyframe_id)

    def erase_observations_in_map(self, keyframe_id, mp_ids):
        '''
        Remove an obeservation for a MapPoint in the map.
        '''
        for mp_id in mp_ids:
            self.msp_map_points[mp_id].erase_observation(keyframe_id)
    
    def get_observations_in_map(self, mp_ids):
        '''
        Remove the obeservation lists for a MapPoint in the map.
        '''
        observations_dict = {}
        for mp_id in mp_ids:
            observed_keyframe_ids = self.msp_map_points[mp_id].get_observations().keys()
            observations_dict[mp_id] = observed_keyframe_ids
        return observations_dict

    def erase_key_frame(self, keyframe):
        """
        Removes a KeyFrame from the map.
        :param keyframe: KeyFrame object to remove.
        """
        self.msp_key_frames.discard(keyframe)

    def set_reference_map_points(self, reference_map_points):
        """
        Sets the reference MapPoints.
        :param reference_map_points: List of MapPoint objects.
        """
        self.mvp_reference_map_points = reference_map_points

    def inform_new_big_change(self):
        """
        Updates the index for a significant map change.
        """
        self.mn_big_change_idx += 1

    def get_last_big_change_idx(self):
        """
        Retrieves the last big change index.
        :return: Integer representing the last big change index.
        """
        return self.mn_big_change_idx

    def get_all_key_frames(self):
        """
        Retrieves all KeyFrames in the map.
        :return: List of KeyFrame objects.
        """
        with self.mutex_map:
            return self.msp_key_frames.copy()

    def get_all_map_points(self):
        """
        Retrieves all MapPoints in the map.
        :return: List of MapPoint objects.
        """
        with self.mutex_map:
            return self.msp_map_points.copy()
    
    def get_all_map_points_for_saving(self):
        map_points_pos = []
        for value in self.msp_map_points.values():
            pos = value.get_world_pos().reshape(-1,1)
            map_points_pos.append(pos)

        return map_points_pos

    def get_reference_map_points(self):
        """
        Retrieves the reference MapPoints.
        :return: List of reference MapPoint objects.
        """
        return self.mvp_reference_map_points

    def map_points_in_map(self):
        """
        Counts the number of MapPoints in the map.
        :return: Integer count of MapPoints.
        """
        with self.mutex_map:
            return len(self.msp_map_points)
    
    def get_map_point(self, idx):
        with self.mutex_map:
            return self.msp_map_points.get(idx)

    def key_frames_in_map(self):
        """
        Counts the number of KeyFrames in the map.
        :return: Integer count of KeyFrames.
        """
        return len(self.msp_key_frames)

    def get_max_kf_id(self):
        """
        Retrieves the maximum KeyFrame ID in the map.
        :return: Integer representing the maximum KeyFrame ID.
        """
        return self.mn_max_kf_id

    def clear(self):
        """
        Clears all data in the map.
        """
        self.msp_map_points.clear()
        self.msp_key_frames.clear()
        self.mvp_reference_map_points.clear()
        self.mvp_key_frame_origins.clear()
        self.mn_max_kf_id = 0
        self.mn_big_change_idx = 0
