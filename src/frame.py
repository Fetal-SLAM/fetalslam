import numpy as np
from copy import deepcopy
import cv2

import logging as _logging

_logger = _logging.getLogger(__name__)

class Frame_index:
    mnNextID = 0
    def get_current_id():
        cur_id = Frame_index.mnNextID
        Frame_index.update_id()
        return cur_id
    def update_id():
        Frame_index.mnNextID += 1

class CameraFrame():
    # Static class attributes
    mfGridElementWidthInv = None
    mfGridElementHeightInv = None
    fx = None
    fy = None
    cx = None
    cy = None
    invfx = None
    invfy = None
    mnMinX = None
    mnMaxX = None
    mnMinY = None
    mnMaxY = None
    mbInitialComputations = False

    frame_grid_rows = 72
    frame_grid_cols = 72

    def __init__(self, image, timestamp, K, disCoef, roi_margin = 0.01):
        self.image = image
        self.timestamp = timestamp

        self.frame_id = Frame_index.get_current_id()

        self.mTcw = np.identity(4)

        # the features from the xfeat, they are tensor
        self.features = None
        
        # only undistort the fake features
        self.undist_features = None

        self.map_points = None

        self.KeysUn_dict = {}

        
        self.reference_kf = None
        self.feature_length = 0

        self.width = image.shape[1]
        self.height = image.shape[0]
        self.roi_margin = int(roi_margin*self.width)

        self.roi = [[self.roi_margin, self.roi_margin], [self.height-self.roi_margin, self.roi_margin], [self.height-self.roi_margin, self.width-self.roi_margin], [self.roi_margin, self.width-self.roi_margin]]
        self.H21 = None

        self.Rcw = None
        self.Rwc = None
        self.tcw = None
        self.Ow = None

        self.mvpMapPoints = None
        self.vbTriangulated = None
        self.mvbOutlier = False
        self.valid_features = 0.0

        self.iou_with_ref = None
        self.intersection_with_ref = None
        self.exclude_area_with_ref = None
        self.new_include_area_with_ref = None

        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.invfx = 1.0 / self.fx
        self.invfy = 1.0 / self.fy

        self.mb = 0.0
        

        if self.mbInitialComputations:
            self.set_properties_from_features()
            
    def setpose(self, Rot, t):
        self.Rcw = Rot
        if np.shape(t) == (3,1):
            self.tcw = t
        else:
            self.tcw = t.reshape((-1,1))
        
        # Tcw is world respect to camera, as the mappoints are fixed and the camera is moving
        self.mTcw[:3,:3] = Rot
        self.mTcw[:3, 3] = t.reshape(-1)
    
    def setpose_T(self, T):
        self.mTcw = T

    def get_rot(self):
        return self.Rcw.copy()
    
    def get_translation(self):
        return self.tcw.reshape(-1).copy()
    
    def get_pose(self):
        return self.mTcw.copy()
    
    def get_vbTriangulated(self):
        return self.vbTriangulated.copy()
    
    def get_mvpMapPoints(self):
        return self.mvpMapPoints.copy()
    
    def get_frame_id(self):
        frame_id = deepcopy(self.frame_id)
        return frame_id

    def set_features_from_xfeat(self, xfeature):
        self.features = xfeature.copy()

    def set_map_points(self, map_points):
        self.map_points = map_points.copy()

    def get_map_points(self):
        if self.map_points is None:
            return None
        else:
            return self.map_points.copy()
        
    def get_KeysUn_dict(self):
        return self.KeysUn_dict.copy()
        

    def set_vbTriangulated(self, vbTriangulated):
        self.vbTriangulated = vbTriangulated.copy()

    def set_undist_features(self, undist_feature_points, H):
        self.H21 = H.copy()
        self.undist_features = undist_feature_points.copy()
        self.set_properties_from_features()

        # check in the new frame, how many features still left in the image
        track_point_flag, inside_points = self.tracked_mappoints()
        
        if track_point_flag:
            self.mvTrackedPoints = inside_points
            num_points_inside = np.sum(inside_points)
            self.valid_features = num_points_inside


    def update_undist_features_in_KeysUn_dict(self):
        dict_key_list = list(self.KeysUn_dict.keys())
        for idx, key in enumerate(dict_key_list):
            self.KeysUn_dict[key] = self.undist_features[idx][0]


    def set_properties_from_features(self):
        if self.undist_features is not None:
            self.feature_length = self.undist_features.shape[0]


            self.mvpMapPoints = {}
            self.KeysUn_dict = {}
            self.vbTriangulated = [False] * self.feature_length
            self.mvbOutlier = [False] * self.feature_length
            self.mvTrackedPoints = np.full((self.feature_length, 1), True, dtype=bool)

            self.mvuRight = [-1] * self.feature_length
            self.mvDepth = [-1] * self.feature_length


    def warp_roi_with_ref_kf(self, H, x_offset = 0):
        self.H21 = H
        cur_roi = np.array(self.roi, dtype='float32').reshape(-1,1,2)

        warped_points_np = cv2.perspectiveTransform(cur_roi, H).reshape(-1, 2)
        warped_points_np[:, 0] += x_offset
        self.ref_roi = warped_points_np.astype(int).tolist()
        iou, intersaction, non_overlap1, non_overlap2 = self.calculate_iou(self.ref_roi, self.roi)

        # update the intersaction information
        self.iou_with_ref = iou
        self.intersection_with_ref = intersaction
        self.exclude_area_with_ref = non_overlap1
        self.new_include_area_with_ref = non_overlap2

        return self.iou_with_ref
    
    def get_visible_features(self):
        '''
        Return the number of visible features in the frame.
        '''
        return self.valid_features


    def calculate_iou(self, rect1, rect2):
        """
        Calculate the IoU between two rectangles defined by their corners.

        Parameters:
        - rect1: List of 4 corners [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        - rect2: List of 4 corners [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        Returns:
        - IoU: Intersection over Union value
        """
        # Extract bounding boxes from corners
        x1_min = min([pt[0] for pt in rect1])
        y1_min = min([pt[1] for pt in rect1])
        x1_max = max([pt[0] for pt in rect1])
        y1_max = max([pt[1] for pt in rect1])

        if x1_min < self.roi_margin:
            x1_min = self.roi_margin
        if y1_min < self.roi_margin:
            y1_min = self.roi_margin
        if x1_max > self.width-self.roi_margin:
            x1_max = self.width-self.roi_margin
        if y1_max > self.height-self.roi_margin:
            y1_max = self.height-self.roi_margin


        x1_1, y1_1, x2_1, y2_1 = (x1_min, y1_min, x1_max, y1_max)

        x2_min = min([pt[0] for pt in rect2])
        y2_min = min([pt[1] for pt in rect2])
        x2_max = max([pt[0] for pt in rect2])
        y2_max = max([pt[1] for pt in rect2])

        if x2_min < self.roi_margin:
            x2_min = self.roi_margin
        if y2_min < self.roi_margin:
            y2_min = self.roi_margin
        if x2_max > self.width-self.roi_margin:
            x2_max = self.width-self.roi_margin
        if y2_max > self.height-self.roi_margin:
            y2_max = self.height-self.roi_margin


        x1_2, y1_2, x2_2, y2_2 = (x2_min, y2_min, x2_max, y2_max)


        # Calculate intersection rectangle
        xi1 = max(x1_min, x2_min)
        yi1 = max(y1_min, y2_min)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)

        if xi1 < xi2 and yi1 < yi2:
            intersection = [xi1, yi1, xi2, yi2]
            inter_area = (xi2 - xi1) * (yi2 - yi1)
        else:
            intersection = []
            inter_area = 0

        # Compute areas of both rectangles
        rect1_area = (x1_max - x1_min) * (y1_max - y1_min)
        rect2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Compute union area
        union_area = rect1_area + rect2_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area

        # Compute non-overlapping regions as coordinates
        non_overlap_coords_1 = []
        non_overlap_coords_2 = []

        if iou >= 0.1 and iou < 0.95:
            # Non-overlapping parts of rect1
            if x1_1 < xi1 - self.roi_margin:
                non_overlap_coords_1.append([x1_min, y1_1, xi1, y2_1])  # Left part of rect1
            if x2_1 > xi2 + self.roi_margin:
                non_overlap_coords_1.append([xi2, y1_1, x2_1, y2_1])  # Right part of rect1
            if y1_1 < yi1 - self.roi_margin:
                non_overlap_coords_1.append([xi1, y1_1, xi2, yi1])  # Top part of rect1
            if y2_1 > yi2 + self.roi_margin:
                non_overlap_coords_1.append([xi1, yi2, xi2, y2_1])  # Bottom part of rect1

            # Non-overlapping parts of rect2
            if x1_2 < xi1 - self.roi_margin:
                non_overlap_coords_2.append([x1_2, y1_2, xi1, y2_2])  # Left part of rect2
            if x2_2 > xi2 + self.roi_margin:
                non_overlap_coords_2.append([xi2, y1_2, x2_2, y2_2])  # Right part of rect2
            if y1_2 < yi1 - self.roi_margin:
                non_overlap_coords_2.append([xi1, y1_2, xi2, yi1])  # Top part of rect2
            if y2_2 > yi2 + self.roi_margin:
                non_overlap_coords_2.append([xi1, yi2, xi2, y2_2])  # Bottom part of rect2
        elif iou < 0.1:
            # If no intersection, the entire rectangles are non-overlapping
            non_overlap_coords_1.append(rect1)
            non_overlap_coords_2.append(rect2)

        return iou, intersection, non_overlap_coords_1, non_overlap_coords_2


    def AssignFeaturesToGrid(self):
        if self.undist_features is not None:
            # Initialize grid with reserved space
            self.mGrid = [[[] for _ in range(self.frame_grid_rows)] for _ in range(self.frame_grid_cols)]
            for i in range(self.frame_grid_cols):
                for j in range(self.frame_grid_rows):
                    self.mGrid[i][j] = []  # Python lists grow dynamically, reserve not needed
            
            # Assign features to the grid
            for i in range(self.feature_length):
                kp = self.undist_features[i]
                n_grid_pos_x, n_grid_pos_y = None, None
                
                if self.PosInGrid(kp, n_grid_pos_x, n_grid_pos_y):
                    self.mGrid[n_grid_pos_x][n_grid_pos_y].append(i)


    def PosInGrid(self, kp):
        """
        Compute the grid position of a keypoint.
        """
        posX = int((kp[0] - self.mnMinX) * self.mfGridElementWidthInv)
        posY = int((kp[1] - self.mnMinY) * self.mfGridElementHeightInv)

        if posX < 0 or posX >= self.frame_grid_cols or posY < 0 or posY >= self.frame_grid_rows:
            return None, None
        return posX, posY

    def update_pose_matrices(self):
        """
        Update the pose matrices for the frame.
        """
        self.Rcw = self.mTcw[:3, :3]  # Extract the rotation matrix (3x3)
        self.Rwc = self.Rcw.T        # Compute the transpose of the rotation matrix
        self.tcw = self.mTcw[:3, 3]  # Extract the translation vector (3x1)
        self.Ow = -np.dot(self.Rwc, self.tcw)  # Compute the world origin in camera coordinates


    def tracked_mappoints(self):
        """
        Label points as True (inside) or False (outside) based on their position in the image.
        :return: A bool flag, A boolean array where True = inside, False = outside.
        """
        if self.H21 is not None and self.undist_features is not None:
            x1 = self.roi_margin
            y1 = self.roi_margin
            x2 = self.height-self.roi_margin
            y2 = self.width-self.roi_margin

            undist_features = self.undist_features.reshape(-1,2).copy()
            x = undist_features[:, 0]
            y = undist_features[:, 1]

            # Check bounds using NumPy
            inside_x = (x > x1) & (x < x2)
            inside_y = (y > y1) & (y < y2)
            inside = inside_x & inside_y  # Combine both conditions
            result = inside.reshape(-1, 1)
            # Return the result as a N, 1 array
            return True, result
        else:
            if self.H21 is None:
                _logger.debug('%s', "[Frame] the current frame doesn't have H21")
            if self.undist_features is None:
                _logger.debug('%s', "[Frame] the current frame doesn't have undist_features")
            return False, None
