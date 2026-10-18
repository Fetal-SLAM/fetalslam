from .frame import CameraFrame
from .visual_reconstruction import VisualReconstruction
from copy import deepcopy
from .features.detector import Xfeat_Det_Matching


import numpy as np
import threading
from collections import OrderedDict

import logging as _logging

_logger = _logging.getLogger(__name__)


class Key_frame_index:
    mnNextID = 0
    def get_current_id():
        cur_id = Key_frame_index.mnNextID
        Key_frame_index.update_id()
        return cur_id
    def update_id():
        Key_frame_index.mnNextID += 1

class KeyFrame:
    def __init__(self, frame:CameraFrame, pMap = None, pKFDB = None, Kf = None):
        # Initialize basic attributes
        self.mnFrameId = frame.get_frame_id()
        self.mTimeStamp = deepcopy(frame.timestamp)
        self.mnGridCols = frame.frame_grid_cols
        self.mnGridRows = frame.frame_grid_rows
        self.mfGridElementWidthInv = frame.mfGridElementWidthInv
        self.mfGridElementHeightInv = frame.mfGridElementHeightInv

        self.fx = frame.fx
        self.fy = frame.fy
        self.cx = frame.cx
        self.cy = frame.cy
        self.invfx = frame.invfx
        self.invfy = frame.invfy

        self.N = frame.feature_length
        self.image = frame.image

        # undistort features
        # only undistort the fake features
        self.mvKeysUn = frame.undist_features

        self.KeysUn_dict = frame.KeysUn_dict

        self.roi_margin = frame.roi_margin
        self.height = frame.height
        self.width = frame.width

        self.xfeat_features = None


        self.mnScaleLevels = 1.0

        self.mnMinX = frame.mnMinX
        self.mnMinY = frame.mnMinY
        self.mnMaxX = frame.mnMaxX
        self.mnMaxY = frame.mnMaxY

        self.mvpMapPoints = frame.mvpMapPoints.copy()
        self.mvbOutlier = [False] * self.N
        self.vbTriangulated = frame.get_vbTriangulated()
        self.mvuRight = frame.mvuRight.copy()
        self.mvDepth = frame.mvDepth.copy()

        self.mvTrackedPoints = frame.mvTrackedPoints.copy()
        self.valid_features = frame.valid_features.copy()

        # only undistort fake feature
        self.new_mvKeysUn = None


        self.mbFirstConnection = True
        self.mbNotErase = False
        self.mbToBeErased = False
        self.mbBad = False
        self.mHalfBaseline = frame.mb / 2
        self.mTcwGBA = None


        self.mnId = Key_frame_index.get_current_id()

        self.mMutexPose = threading.RLock()
        self.mMutexConnections = threading.RLock()
        self.mMutexFeatures = threading.RLock()
        self.mConnectedKeyFrameWeights = {}
        self.mvpOrderedConnectedKeyFrames = []
        self.mvOrderedWeights = []
        self.Ow = None
        self.Twc = None
        self.Cw = None

        # Set pose
        self.set_pose(frame.get_pose())


        self.mspChildrens = set()
        self.mpParent = None
        self.mbBad = False

    def get_keyframe_id(self):
        return self.mnId

    def set_pose(self, Tcw_):
        # Tcw is world respect to camera, as the mappoints are fixed and the camera is moving
        with self.mMutexPose:
            self.Tcw = Tcw_.copy()
            self.Rcw = self.Tcw[:3, :3]
            self.tcw = self.Tcw[:3, 3].reshape(-1,1)
            self.Rwc = self.Rcw.T
            self.Ow = -np.dot(self.Rwc, self.tcw)

            self.Twc = np.eye(4, dtype=Tcw_.dtype)
            self.Twc[:3, :3] = self.Rwc
            self.Twc[:3, 3] = self.Ow.reshape(-1)
            center = np.array([self.mHalfBaseline, 0, 0, 1], dtype=float)
            self.Cw = np.dot(self.Twc, center)

    def get_pose(self):
        with self.mMutexPose:
            return self.Tcw.copy()

    def get_pose_inverse(self):
        with self.mMutexPose:
            return self.Twc.copy()

    def get_rot(self):
        with self.mMutexPose:
            return self.Rcw.copy()

    def get_translation(self):
        with self.mMutexPose:
            return self.tcw.reshape(-1).copy()

    def get_camera_center(self):
        with self.mMutexPose:
            return self.Ow.copy()

    def get_stereo_center(self):
        with self.mMutexPose:
            return self.Cw.copy()

    def compute_scene_median_depth(self, q):
        """
        Compute the median scene depth for the MapPoints visible from this KeyFrame.

        Parameters:
        q (int): Divisor to select the quantile of the depth values.

        Returns:
        float: The median depth of the scene.
        """
        vpMapPoints = [value for value in self.mvpMapPoints.values()]
        Tcw_ = self.Tcw.copy()

        vDepths = []

        # Extract the rotation and translation components
        Rcw2 = Tcw_[:3, :3][2, :].reshape(1, -1)  # Extract the 3rd row of rotation matrix
        zcw = Tcw_[2, 3]  # Translation component in Z direction

        for pMP in vpMapPoints:
            if pMP:
                x3Dw = pMP.get_world_pos()
                z = np.dot(Rcw2, x3Dw) + zcw
                vDepths.append(z)

        # Sort the depths
        vDepths.sort()

        # Return the quantile depth
        return vDepths[(len(vDepths) - 1) // q] if vDepths else 0.0

    def set_new_keyfeatures(self, features):
        self.new_mvKeysUn = features.copy()

    def get_new_keyfeatures(self):
        return self.new_mvKeysUn.copy()

    def set_xfeat_features(self, xfeat_features):
        self.xfeat_features = xfeat_features.copy()

    def get_xfeat_features(self):
        return self.xfeat_features

    def get_vbTriangulated(self):
        return self.vbTriangulated.copy()

    def get_mvbOutlier(self):
        return self.mvbOutlier.copy()

    def add_map_point(self, pMP, idx, feature_index=None):
        feature_index = idx if feature_index is None else feature_index
        if 0 <= feature_index < self.N:
            feature = self.mvKeysUn[feature_index][0].copy()
            self.KeysUn_dict[idx] = feature
            self.mvpMapPoints[idx] = pMP

    def compact_features_to_map_points(self):
        """Keep dense feature arrays aligned with map-point dictionary order."""
        with self.mMutexFeatures:
            features = [
                np.asarray(self.KeysUn_dict[map_point_id], dtype=np.float32)
                for map_point_id in self.mvpMapPoints
                if map_point_id in self.KeysUn_dict
            ]
            if features:
                self.mvKeysUn = np.asarray(features, dtype=np.float32).reshape(-1, 1, 2)
            else:
                self.mvKeysUn = np.empty((0, 1, 2), dtype=np.float32)

            self.N = len(features)
            self.mvbOutlier = [False] * self.N
            self.vbTriangulated = [True] * self.N
            self.mvTrackedPoints = np.ones((self.N, 1), dtype=bool)
            self.mvuRight = [-1] * self.N
            self.mvDepth = [-1] * self.N
            self.valid_features = self.N

    def update_map_point(self, pMP, feature, idx):
        if pMP is None:
            return False
        with self.mMutexFeatures:
            existing_keys = list(self.KeysUn_dict)
            is_new = idx not in self.KeysUn_dict
            self.KeysUn_dict[idx] = np.asarray(feature).copy()
            self.mvpMapPoints[idx] = pMP
            if is_new:
                new_feature = np.asarray(feature, dtype=np.float32).reshape(1, 1, 2)
                self.mvKeysUn = np.vstack((self.mvKeysUn, new_feature))
                self.mvbOutlier.append(False)
                self.vbTriangulated.append(True)
                self.mvTrackedPoints = np.vstack((self.mvTrackedPoints, [[True]]))
            else:
                position = existing_keys.index(idx)
                self.mvKeysUn[position][0] = feature
            self.N = len(self.KeysUn_dict)
        return True

    def delete_map_point_match(self, idx):
        with self.mMutexFeatures:
            keys = list(self.KeysUn_dict)
            position = keys.index(idx) if idx in self.KeysUn_dict else None
            point_removed = self.mvpMapPoints.pop(idx, None) is not None
            feature_removed = self.KeysUn_dict.pop(idx, None) is not None
            if position is not None:
                self.mvKeysUn = np.delete(self.mvKeysUn, position, axis=0)
                if position < len(self.mvbOutlier):
                    self.mvbOutlier.pop(position)
                if position < len(self.vbTriangulated):
                    self.vbTriangulated.pop(position)
                if position < len(self.mvTrackedPoints):
                    self.mvTrackedPoints = np.delete(self.mvTrackedPoints, position, axis=0)
            self.N = len(self.KeysUn_dict)
        return point_removed or feature_removed

    def get_KeysUn_dict_feature(self, idx):
        with self.mMutexFeatures:
            feature = self.KeysUn_dict.get(idx)
            return None if feature is None else feature.copy()

    def get_mvKeysUn(self):
        if len(self.mvKeysUn) != self.N:
            # use KeysUn_dict generate mvKeysUn
            self.mvKeysUn = np.zeros((self.N, 1, 2), dtype=np.float32)
            for i, key in enumerate(self.KeysUn_dict.keys()):
                feature = self.KeysUn_dict[key]
                self.mvKeysUn[i][0] = feature

        return self.mvKeysUn.copy()


    def get_KeysUn_dict(self):
        return self.KeysUn_dict.copy()

    def get_img(self):
        if self.image is None:
            return None
        else:
            return deepcopy(self.image)

    def get_feature_colors(self):
        # Assuming the image is in BGR format
        colors = np.zeros((self.N, 3), dtype=np.uint8)

        mvKeysUn_list = self.KeysUn_dict.keys()
        # Extract colors at the given points
        for i, feature_key in enumerate(mvKeysUn_list):
            feature = self.KeysUn_dict[feature_key]
            x, y = int(feature[0]), int(feature[1])
            if x < self.width and x > 0 and y < self.height and y > 0:
                colors[i] = self.image[y, x]  # Note: OpenCV uses row (y), column (x)
            else:
                colors[i] = np.array((255,255,255))


        # Convert to RGB if needed
        colors_rgb = colors[:, ::-1]
        return colors_rgb

    def get_feature_color(self, feature_2d):
        x,y = int(feature_2d[0]), int(feature_2d[1])
        color = self.image[y, x]
        # Convert to RGB if needed
        colors_rgb = color[::-1]
        return colors_rgb.copy()


    def get_mvpMapPoints(self):
        with self.mMutexFeatures:
            return self.mvpMapPoints.copy()

    def replace_map_point_match(self, idx, pMP):
        self.mvpMapPoints[idx] = pMP

    def get_map_point(self, idx):
        with self.mMutexFeatures:
            return self.mvpMapPoints.get(idx)

    def get_map_points(self):
        return self.mvpMapPoints.values()

    def set_map_points(self, mappoints):
        self.map_points = mappoints

    def tracked_map_points(self, min_obs):
        n_points = 0
        for value in self.mvpMapPoints.values():
            pMP = value
            if pMP and not pMP.is_bad() and (pMP.Observations() >= min_obs if min_obs > 0 else True):
                n_points += 1
        return n_points

    def get_map_point_matches(self):
        return [value for value in self.mvpMapPoints.values()]


    def update_after_triangulation(self, new_map_points):

        new_points_count = len(new_map_points)

        success_count = 0
        need_to_be_delete = []
        # update follow information
        for idx in range(new_points_count):
            new_mp = new_map_points[idx]
            if new_mp != None:
                mp_id = new_mp.mnId
                feature = self.new_mvKeysUn[idx][0].copy()
                self.mvpMapPoints[mp_id] = new_mp
                self.KeysUn_dict[mp_id] = feature
                self.mvbOutlier.append([False])
                self.vbTriangulated.append([True])
                success_count += 1
            else:
                self.vbTriangulated.append([False])
                need_to_be_delete.append(idx)

            self.mvuRight.append([-1])
            self.mvDepth.append([-1])


        self.valid_new_mvKeysUn = np.delete(self.new_mvKeysUn, need_to_be_delete, axis=0)

        new_tracked_points = np.full((success_count, 1), True, dtype=bool)
        self.mvTrackedPoints = np.vstack((self.mvTrackedPoints, new_tracked_points))
        self.valid_features += success_count

        # Concatenate the old features with new features
        self.mvKeysUn = np.vstack((self.mvKeysUn, self.valid_new_mvKeysUn))


        self.N = len(self.mvKeysUn)
        if self.N == len(self.mvpMapPoints):
            _logger.debug('%s', "update keyframe after trianglation successfully!")
        else:
            _logger.debug('%s', f"self.N is {self.N}, len(self.mvpMapPoints) is {len(self.mvpMapPoints)}")
            _logger.debug('%s', "check keyframe features number after trianglation!")


    def update_features_and_mappoints(self):
        mask = self.mvTrackedPoints.copy()

        update_mvbOutlier = []
        update_vbTriangulated = []
        update_mvKeysUn = []
        filtered_mvpMapPoints = {}
        filtered_KeysUn_dict = {}

        dict_keys = list(self.mvpMapPoints.keys())
        for i, (keep, dict_key) in enumerate(zip(mask.reshape(-1), dict_keys)):
            if keep:
                filtered_mvpMapPoints[dict_key] = self.mvpMapPoints[dict_key]
                filtered_KeysUn_dict[dict_key] = self.KeysUn_dict[dict_key]
                update_mvbOutlier.append(self.mvbOutlier[i])
                update_vbTriangulated.append(self.vbTriangulated[i])
                update_mvKeysUn.append(self.mvKeysUn[i])
            else:
                self.mvpMapPoints[dict_key].erase_observation(self.mnId)

        self.mvKeysUn = np.asarray(update_mvKeysUn).copy()
        self.mvbOutlier = update_mvbOutlier.copy()
        self.vbTriangulated = update_vbTriangulated.copy()
        self.mvpMapPoints = filtered_mvpMapPoints.copy()
        self.KeysUn_dict = filtered_KeysUn_dict.copy()

        self.N = self.mvKeysUn.shape[0]

        # update the tracked points
        update_mvTrackedPoints = mask[mask].reshape(-1,1)
        self.mvTrackedPoints = update_mvTrackedPoints.copy()


        if self.N == len(self.mvpMapPoints):
            _logger.debug('%s', "update keyframe after trianglation successfully!")
        else:
            _logger.debug('%s', f"self.N is {self.N}, len(self.mvpMapPoints) is {len(self.mvpMapPoints)}")
            _logger.debug('%s', "check keyframe features number after trianglation!")


    def tracked_mappoints(self, rect):
        """
        Label points as True (inside) or False (outside) based on their position in the image.
        Parameters:
        - rect: List of 2 corners [x1,y1,x2,y2]

        return: A bool flag, A boolean array where True = inside, False = outside.
        """
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]

        undist_features = self.mvKeysUn.reshape(-1,2).copy()
        x = undist_features[:, 0]
        y = undist_features[:, 1]

        # Check bounds using NumPy
        inside_x = (x > x1) & (x < x2)
        inside_y = (y > y1) & (y < y2)
        inside = inside_x & inside_y  # Combine both conditions
        result = inside.reshape(-1, 1)
        # Return the result as a N, 1 array
        return result.copy()


    def set_mvTrackedPoints(self, inliers):
        mask = inliers.copy()
        update_mvbOutlier = []
        update_vbTriangulated = []
        update_mvKeysUn = []
        filtered_mvpMapPoints = {}
        filtered_KeysUn_dict = {}

        dict_keys = list(self.mvpMapPoints.keys())

        for i, (keep, dict_key) in enumerate(zip(mask.reshape(-1), dict_keys)):
            if keep:
                filtered_mvpMapPoints[dict_key] = self.mvpMapPoints[dict_key]
                filtered_KeysUn_dict[dict_key] = self.KeysUn_dict[dict_key]
                update_mvbOutlier.append(self.mvbOutlier[i])
                update_vbTriangulated.append(self.vbTriangulated[i])
                update_mvKeysUn.append(self.mvKeysUn[i])


        self.mvKeysUn = np.asarray(update_mvKeysUn).astype(int).copy()
        self.mvbOutlier = update_mvbOutlier.copy()
        self.vbTriangulated = update_vbTriangulated.copy()
        self.mvpMapPoints = filtered_mvpMapPoints.copy()
        self.KeysUn_dict = filtered_KeysUn_dict.copy()

        self.N = self.mvKeysUn.shape[0]


        # update the tracked points
        update_mvTrackedPoints = mask[mask].reshape(-1,1)

        self.mvTrackedPoints = update_mvTrackedPoints.copy()

        self.valid_features = np.sum(inliers)


class KeyFrameType:
    GrandParent = 0
    Parent = 1
    Child = 2

from scipy.spatial import cKDTree

class TrajectoryKDTree:
    def __init__(self):
        self.points = []
        self.points_id = []
        self.kdtree = None

    def add_point(self, point, point_id):
        """Add a 3D point and rebuild the KD-tree."""
        self.points.append(point)
        self.kdtree = cKDTree(self.points)
        self.points_id.append(point_id)

    def update_all_points(self, new_points_ids:dict):
        new_points = []
        new_points_id = []
        for ids, new_point in new_points_ids.items():
            new_point_pos = new_point[:3, 3].reshape(-1)
            new_points.append(new_point_pos)
            new_points_id.append(ids)

        self.points = new_points
        self.points_id = new_points_id
        self.kdtree = cKDTree(self.points)

    def query_k_nearest(self, query_point, k=1):
        """Query the k nearest neighbors to the given 3D point."""
        if self.kdtree is None or len(self.points) < k:
            return [], [], []
        distances, indices = self.kdtree.query(query_point, k=k)
        if k == 1:
            return [self.points_id[indices]], [self.points[indices]], [distances]
        return [self.points_id[i] for i in indices], [self.points[i] for i in indices], distances


    def query_k_nearest_exclude_nearby_ids(self, query_point, query_id, k=3, id_margin=5, extra=10):
        """
        Query k nearest neighbors excluding points with IDs close to query_id.
        `id_margin` defines how close is considered "too close".
        `extra` allows fetching more neighbors in case many get filtered out.
        """
        if self.kdtree is None or len(self.points) == 0:
            return [], [], []

        # Query more neighbors than needed to have buffer for filtering
        num_neighbors = min(k + extra, len(self.points))
        dists, indices = self.kdtree.query(query_point, k=num_neighbors)

        if np.isscalar(indices):
            indices = [indices]
            dists = [dists]

        valid_neighbors = []
        valid_dists = []
        valid_ids = []

        for idx, dist in zip(indices, dists):
            pid = self.points_id[idx]
            if abs(pid - query_id) > id_margin:
                valid_neighbors.append(self.points[idx])
                valid_dists.append(dist)
                valid_ids.append(pid)
            if len(valid_neighbors) == k:
                break

        return valid_ids, valid_neighbors, valid_dists

class KeyFrameNode():
    def __init__(self, name, kf:KeyFrame, H_ref_cur, frame_type = KeyFrameType.Child, parent_name = None, child_counters = 0):
        """
        Initialize a node with a name and an empty list of children.
        """
        self.name = name
        self.kf = kf
        self.homography_from_parent_to_cur = H_ref_cur
        self.type = frame_type
        self.child_counters = 0
        self.parent_name = None
        if self.type == KeyFrameType.Parent:
            self.child_counters = child_counters  # Counters for child indexing per parent

        elif self.type == KeyFrameType.Child:
            self.parent_name = parent_name

    def get_H_ref_cur(self):
        """
        Get the Homography matrix H_cur_ref.

        return: numpy.array (3x3)
        """
        return self.homography_from_parent_to_cur.copy()


    def set_H_ref_cur(self, new_H_ref_cur):
        self.homography_from_parent_to_cur = new_H_ref_cur

    def get_kf(self):
        return self.kf

    def get_frame_type(self):
        return self.type

class KeyFrameTree():
    def __init__(self, max_parents = 5, iou_threshold = 0.95, gp_promote_threshold = 0.8):
        """
        Initialize the hierarchy with a dictionary of parent nodes
        and transformation matrices to describe relationships between parents.
        """
        self.parents = OrderedDict()  # Maintain order of parent addition
        self.max_parents = max_parents
        self.newest_parent_name = None

        self.grand_parents = OrderedDict()  # Maintain order of parent addition
        self.grand_parent_transforms = {}  # Store parent-to-parent transformations
        self.newest_grand_parent_name = None

        self.grand_parents_kdtree = TrajectoryKDTree()

        self.repeat_iou_threshold = iou_threshold

        self.grand_parent_promote_threshold = gp_promote_threshold

        self.new_parent_flag = False

        # for colmap result
        self.parent_node_store_for_colmap = {}


    def set_xfeat(self, xfeat:Xfeat_Det_Matching):
        self.xfeat = xfeat

    def set_recon(self, recon:VisualReconstruction):
        self.vis_recon = recon


    ''' ----------------------- operation to grandparent node -----------------------'''

    def add_grand_parent(self, kf):
        """
        Add a grand parent node to the hierarchy. And reset the parameters

        Args:
            parent (str): Name of the first parent node.
        """
        grand_parent_name = kf.mnId
        # Extract translation distance from the transformation matrix
        # Promote parent1 to a grandparent
        self.grand_parents[grand_parent_name] = KeyFrameNode(grand_parent_name, kf, np.identity(3), frame_type=KeyFrameType.GrandParent)
        global_pose = kf.get_translation()
        self.newest_grand_parent_name = grand_parent_name
        self.grand_parents_kdtree.add_point(global_pose, grand_parent_name)
        _logger.debug('%s', f"Promoted a new grandparent {grand_parent_name}.")


    def check_if_ok_to_promote_parent_node(self):
        # get all parents, compare with the latest grandparents
        grand_parent_xfeat = self.grand_parents[self.newest_grand_parent_name].kf.xfeat_features
         # check if the lastest node moves back
        for parent_name in self.parents.keys():
            other_parent_xfeat = self.parents[parent_name]["node"].kf.xfeat_features

            if other_parent_xfeat is not None:
                xfeat_flag, H_other_parent_candidate = self.xfeat.match(other_parent_xfeat, grand_parent_xfeat)
                if xfeat_flag:
                    iou = self.vis_recon.compute_iou_from_homography_fast(H_other_parent_candidate)
                    if iou <= self.grand_parent_promote_threshold:
                        if parent_name  not in self.grand_parents:
                            self.add_grand_parent(self.parents[parent_name]["node"].kf)
                            _logger.debug('%s', f"promote id {parent_name} as grand parent.")
                            break


    def add_parent_node_after_close_loop(self, close_loop_grand_parent_name):
        # check target node or close loop node is better for current frame
        new_kf = self.grand_parents[close_loop_grand_parent_name].get_kf()

        self.add_parent(new_kf)


    ''' ----------------------- operation to parent node -----------------------'''

    def add_parent(self, parent_kf, H_ref_cur = np.identity(3)):
        """
        Add a parent node to the hierarchy.

        Args:
            parent_name (str): Name of the parent node.
        """
        parent_name = parent_kf.mnId
        frametype = KeyFrameType.Parent
        if parent_name in self.parents:
            parent_data = self.parents.pop(parent_name)
            parent_data["node"] = KeyFrameNode(
                parent_name, parent_kf, H_ref_cur, frametype
            )
            self.parents[parent_name] = parent_data
        else:
            self.parents[parent_name] = {
                "node": KeyFrameNode(parent_name, parent_kf, H_ref_cur, frametype),
                "children": [],
            }
        self.newest_parent_name = parent_name

        # # get oldest parent image and pose

        # update the parents node
        self.update_node()

    def update_node(self):
        """
        If the number of parents exceeds
        the maximum allowed, delete the oldest parent.

        Args:
            parent_name (str): Name of the parent node.
        """
        # Check and enforce the maximum parent limit
        if len(self.parents) > self.max_parents:
            # Remove the oldest parent
            oldest_parent = next(iter(self.parents))  # Get the first key in the OrderedDict

            del self.parents[oldest_parent]
            _logger.debug('%s', f"Oldest parent '{oldest_parent}' deleted to maintain limit of {self.max_parents} parents.")


    def switch_parent(self, cur_parent_name, new_parent_name, latest_child_name, H_ref_cur):
        if cur_parent_name not in self.parents:
            _logger.debug('%s', f"promote child Parent '{cur_parent_name}' does not exist!")
            return


        for children in self.parents[cur_parent_name]["children"]:
            _logger.debug('%s', f"children is {children} and latest child name is {latest_child_name}")
            if children == latest_child_name:
                target_child_kf = self.parents[cur_parent_name][latest_child_name].get_kf()
                # switch the parent node
                self.parents = self.swap_keys(self.parents, cur_parent_name, new_parent_name)
                self.add_child_to_parent(new_parent_name, target_child_kf, H_ref_cur)

            else:
                del self.parents[cur_parent_name][children]

        self.parents[cur_parent_name]["children"] = []

        self.newest_parent_name = new_parent_name
        _logger.debug('%s', f"switch parent from {cur_parent_name} to {self.newest_parent_name} and add child {latest_child_name} to {new_parent_name}.")

        # update node
        self.update_node()

    def swap_keys(self, odict, key1, key2):
        new_odict = OrderedDict()
        for k, v in odict.items():
            if k == key1:
                continue
            if k == key2:
                new_odict[key2] = v
                new_odict[key1] = odict[key1]
            else:
                new_odict[k] = v
        return new_odict


    ''' ----------------------- operation to child node ----------------------- '''

    def promote_child_to_parent(self, parent_name, child_name):
        """
        Check if a child node satisfies a condition. If yes, promote it to a new parent.

        Args:
            parent_name (str): Name of the parent node.
            child_name (str): Name of the child node to evaluate.
            condition (callable): A function that takes a Node and returns True or False.
        """
        if parent_name not in self.parents:
            _logger.debug('%s', f"promote child Parent '{parent_name}' does not exist!")
            return


        for children in self.parents[parent_name]["children"]:
            if children == child_name:
                target_child = self.parents[parent_name].pop(child_name)
            else:
                del self.parents[parent_name][children]

        child_kf = target_child.get_kf()
        child_kf.update_features_and_mappoints()


        # Promote the child to a new parent
        self.add_parent(child_kf)

        self.parents[parent_name]["children"] = []


    # need to change
    def add_child_to_parent(self, parent_name, child_kf, H_ref_cur, transform_matrix =None):
        """
        Add a child node to a specified parent with a transformation matrix
        and an automatically generated name.

        Args:
            parent_name (str): Name of the parent node.
            transform_matrix (numpy.ndarray): 4x4 transformation matrix.
        """
        if parent_name in self.parents:


            # 2. if exploring the new area, check the new child node with other children, if it is new area is covisible and new enough (IOU < 95)
            if len(self.parents[parent_name]["children"]) > 0:
                last_child_name = self.parents[parent_name]["children"][-1]
                last_child_node = self.parents[parent_name][last_child_name]
                last_child_node_H = last_child_node.get_H_ref_cur()

                H_new_child_cur = np.dot(np.linalg.inv(H_ref_cur), last_child_node_H)
                iou = self.vis_recon.compute_iou_from_homography_fast(H_new_child_cur)

                # 3. if the current child frame is not new enough, delete current one
                if iou > self.repeat_iou_threshold:
                    return False


            # 4. if the new child frame can reconstruct with other frame, waiting localmapping thread
            # Generate the child name based on the parent's index
            self.parents[parent_name]["node"].child_counters += 1
            child_counter = self.parents[parent_name]["node"].child_counters
            child_name = f"Child {parent_name}.{child_counter}"
            child_node = KeyFrameNode(child_name, child_kf, H_ref_cur)

            self.parents[parent_name]["children"].append(child_name)
            self.parents[parent_name][child_name] = child_node

            # calculate the new covisible area IOU between children

            _logger.debug('%s', f"Added {child_name} to {parent_name}.")
            return True
        else:
            _logger.debug('%s', f"Parent '{parent_name}' does not exist!")
            return False


    def remove_child(self, parent_name, child_name):
        if parent_name in self.parents:
            if child_name in self.parents[parent_name]:
                del self.parents[parent_name][child_name]
                self.parents[parent_name]["children"].remove(child_name)


    def remove_last_child(self, parent_name):
        """
        delete the last child in the parent node

        Args:
            parent_name (str): Name of the parent node.
            child_name (str): Name of the child node.
        """
        if parent_name in self.parents:
            child_name = self.parents[parent_name]["children"][-1]
            del self.parents[parent_name][child_name]
            del self.parents[parent_name]["children"][-1]


    #     Args:
    #         parent_name (str): Name of the parent node.
    #         child_name (str): Name of the child node.
    #     """
    #     # reset the child counter in this frame


    #     # copy the lastest child to the first child


    def display(self):
        """
        Display the hierarchy of parents and their children,
        along with the transformation matrices.
        """
        _logger.debug('%s', "Hierarchy:")
        for parent_name, parent_data in self.parents.items():
            _logger.debug('%s', f"\nParent: {parent_name}")
            for child in parent_data["children"]:
                _logger.debug('%s', f"  - {child}")

    def get_all_temp_keyframes(self):
        """
        Get all the temporary keyframes in the hierarchy.

        """
        all_keyframes = []
        for parent_name, parent_data in self.parents.items():
            all_keyframes.append(parent_data["node"].kf)
            for child in parent_data["children"]:
                all_keyframes.append(parent_data[child].kf)
        return all_keyframes

    def get_all_keyframes_for_loopclosing(self):
        """
        Get all the keyframes in the hierarchy.

        """
        all_keyframes = {}

        # get all grand parent nodes
        for grand_parent_name, gp_data in self.grand_parents.items():
            grand_parent_kf = gp_data.get_kf()
            gp_kf_id = grand_parent_kf.get_keyframe_id()
            all_keyframes[gp_kf_id] = grand_parent_kf

        # get all parent nodes
        for parent_name, parent_data in self.parents.items():
            parent_kf = parent_data["node"].get_kf()
            parent_kf_id = parent_kf.get_keyframe_id()
            if parent_kf_id not in all_keyframes:
                all_keyframes[parent_kf_id] = parent_kf

        return all_keyframes

    def get_all_keyframes_for_saving(self):
        """
        Get all the keyframes in the hierarchy.

        """
        all_keyframes = {}

        # get all grand parent nodes
        for grand_parent_name, gp_data in self.grand_parents.items():
            grand_parent_kf = gp_data.get_kf()
            gp_kf_id = grand_parent_kf.get_keyframe_id()
            all_keyframes[gp_kf_id] = { "image":grand_parent_kf.image, "pose":grand_parent_kf.get_pose() }

        # get all parent nodes
        for parent_name, parent_data in self.parents.items():
            parent_kf = parent_data["node"].get_kf()
            parent_kf_id = parent_kf.get_keyframe_id()
            if parent_kf_id not in all_keyframes:
                all_keyframes[parent_name] = { "image":parent_kf.image, "pose":parent_kf.get_pose() }

        return all_keyframes


    def update_all_keyframe_pos(self, optimized_frame_poses):
        """
        Update all the keyframes poses after optimization.

        """
        for parent_name, parent_data in self.parents.items():
            kf_id = parent_data["node"].kf.get_keyframe_id()
            if kf_id in optimized_frame_poses:
                parent_data["node"].kf.set_pose(optimized_frame_poses[kf_id])

            if len(parent_data["children"]) > 0:
                for child_name in parent_data["children"]:
                    child_kf = parent_data[child_name].kf
                    child_kf_id = child_kf.get_keyframe_id()
                    if child_kf_id in optimized_frame_poses:
                        child_kf.set_pose(optimized_frame_poses[child_kf_id])
        return

    def update_keyframe_tree(self, new_reconstruction_flag, best_candidate_name=None):
        # update the keyframe tree (parent node, grandparent node)
        if new_reconstruction_flag:
            # promote the child to parent node and culling child nodes
            self.promote_child_to_parent(self.newest_parent_name, best_candidate_name)
            self.new_parent_flag = True
            # check if anynode is fine for the grandparents node
            self.check_if_ok_to_promote_parent_node()


        return


    def erase_map_points(self, map_points_observations):
        """
        Get all the keyframes in the hierarchy and delete target map point in keyframe

        """

        keyframes_id_class = {}
        # get all grand parent nodes
        for grand_parent_name, grand_parent_data in self.grand_parents.items():

            keyframes_id_class[grand_parent_name] = grand_parent_data.kf


        # get all parent nodes
        for parent_name, parent_data in self.parents.items():

            if parent_name not in keyframes_id_class:
                keyframes_id_class[parent_name] = parent_data["node"].kf

                if len(parent_data["children"]) > 0:
                    # get all child nodes
                    for child_name in parent_data["children"]:
                        if child_name not in keyframes_id_class:
                            child_kf = parent_data[child_name].kf
                            keyframes_id_class[child_name] = child_kf


        for mp_id, obsers in map_points_observations.items():
            for observation_kf_id in obsers:
                if observation_kf_id in keyframes_id_class:
                    keyframes_id_class[observation_kf_id].delete_map_point_match(mp_id)
