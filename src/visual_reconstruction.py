import numpy as np
import cv2
from .runtime_config import runtime_settings

import logging as _logging

_logger = _logging.getLogger(__name__)

class VisualReconstruction():
    def __init__(
        self,
        cam_K_matrix,
        distortion_co=None,
        width=720,
        height=720,
        margin_percent=0.05,
        min_point_parallax_degrees=1.0,
        max_depth_factor=5.0,
        settings=None,
    ):
        self.settings = settings or runtime_settings("reconstruction")
        self.sigma = self.settings.pixel_sigma
        self.sigma2 = self.sigma ** 2
        self.K = cam_K_matrix
        self.distortion_co = distortion_co
        self.width = width
        self.height = height
        self.margin = int(margin_percent * self.width)
        self.min_point_parallax_cos = float(
            np.cos(np.deg2rad(min_point_parallax_degrees))
        )
        self.max_depth_factor = float(max_depth_factor)
        

    def ReconstructH(self, kp1, kp2, H21, minParallax=None, minTriangulated=None):
        '''
        Reconstruct the feature points by triangulation from Homography matrix
        Input: matched_features in previous frame kp1 and in current frame kp2, Homography matrix H21
        Output: Rotation R21, 
                Translation t21, 
                3D location vP3D,
                Flag of if the feature point pair is successfully triangulated vbTriangulated
        
        '''
        minParallax = self.settings.min_parallax_degrees if minParallax is None else minParallax
        minTriangulated = self.settings.min_triangulated if minTriangulated is None else minTriangulated
        # Count the number of inlier matches
        vbMatchesInliers = [cv2.DMatch(i,i,0) for i in range(len(kp1))]
        N = len(vbMatchesInliers)

        # Compute the inverse of K and A
        invK = np.linalg.inv(self.K)
        A = invK @ H21 @ self.K

        # Perform SVD decomposition
        w, U, Vt = cv2.SVDecomp(A, flags=cv2.SVD_FULL_UV)
        V = Vt.T

        # Compute determinant product

        s = np.linalg.det(U) * np.linalg.det(Vt)

        # Extract singular values
        d1, d2, d3 = w.flatten()

        # Validate singular values
        if d1 / d2 < 1.00001 or d2 / d3 < 1.00001:
            return False, None, None, None, None

        # Containers for motion hypotheses
        vR, vt, vn = [], [], []

        # First case: d' = d2
        aux1 = np.sqrt((d1**2 - d2**2) / (d1**2 - d3**2))
        aux3 = np.sqrt((d2**2 - d3**2) / (d1**2 - d3**2))
        x1 = [aux1, aux1, -aux1, -aux1]
        x3 = [aux3, -aux3, aux3, -aux3]

        aux_stheta = np.sqrt((d1**2 - d2**2) * (d2**2 - d3**2)) / ((d1 + d3) * d2)
        ctheta = (d2**2 + d1 * d3) / ((d1 + d3) * d2)
        stheta = [aux_stheta, -aux_stheta, -aux_stheta, aux_stheta]

        for i in range(4):
            Rp = np.eye(3, dtype=np.float32)
            Rp[0, 0] = ctheta
            Rp[0, 2] = -stheta[i]
            Rp[2, 0] = stheta[i]
            Rp[2, 2] = ctheta

            R = s * U @ Rp @ Vt
            vR.append(R)

            tp = np.array([[x1[i]], [0], [-x3[i]]], dtype=np.float32) * (d1 - d3)
            t = U @ tp
            vt.append(t / np.linalg.norm(t))

            np_ = np.array([[x1[i]], [0], [x3[i]]], dtype=np.float32)
            n = V @ np_
            if n[2, 0] < 0:
                n = -n
            vn.append(n)

        # Second case: d' = -d2
        aux_sphi = np.sqrt((d1**2 - d2**2) * (d2**2 - d3**2)) / ((d1 - d3) * d2)
        cphi = (d1 * d3 - d2**2) / ((d1 - d3) * d2)
        sphi = [aux_sphi, -aux_sphi, -aux_sphi, aux_sphi]

        for i in range(4):
            Rp = np.eye(3, dtype=np.float32)
            Rp[0, 0] = cphi
            Rp[0, 2] = sphi[i]
            Rp[1, 1] = -1
            Rp[2, 0] = sphi[i]
            Rp[2, 2] = -cphi

            R = s * U @ Rp @ Vt
            vR.append(R)

            tp = np.array([[x1[i]], [0], [x3[i]]], dtype=np.float32) * (d1 + d3)
            t = U @ tp
            vt.append(t / np.linalg.norm(t))

            np_ = np.array([[x1[i]], [0], [x3[i]]], dtype=np.float32)
            n = V @ np_
            if n[2, 0] < 0:
                n = -n
            vn.append(n)

        # Evaluate all hypotheses
        bestGood = 0
        secondBestGood = 0
        bestSolutionIdx = -1
        bestParallax = -1
        bestP3D = []
        bestTriangulated = []

        for i in range(8):
            # Assume CheckRT is implemented elsewhere
            
            nGood, parallaxi, vP3Di, vbTriangulatedi = self.CheckRT(vR[i], vt[i], kp1, kp2, vbMatchesInliers, self.K, self.settings.reprojection_sigma_factor * self.sigma2)
            if nGood > bestGood:
                secondBestGood = bestGood
                bestGood = nGood
                bestSolutionIdx = i
                bestParallax = parallaxi
                bestP3D = vP3Di
                bestTriangulated = vbTriangulatedi
            elif nGood > secondBestGood:
                secondBestGood = nGood

        # Validate the best solution
        c1 = secondBestGood < self.settings.ambiguity_ratio * bestGood
        c2 = bestParallax >= minParallax
        c3 = bestGood > minTriangulated
        c4 = bestGood > self.settings.min_good_fraction * N

        _logger.debug('%s', f"best is {bestGood}, second best is {secondBestGood}")
        _logger.debug('%s', f"c1 is {c1}, c2 is {c2}, c3 is {c3}, c4 is {c4}")
        
        if c1 and c2 and c3 and c4:
            R21 = vR[bestSolutionIdx]
            t21 = vt[bestSolutionIdx]
            vP3D = bestP3D
            vbTriangulated = bestTriangulated
            return True, R21, t21, vP3D, vbTriangulated

        return False, None, None, None, None


    def CheckRT(self, R, t, vKeys1, vKeys2, vbMatchesInliers, K, th2):
        # Calibration parameters
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        vbGood = [False] * len(vKeys1)
        vP3D = [None] * len(vKeys1)

        vCosParallax = []

        # Camera 1 Projection Matrix K[I|0]
        P1 = np.hstack((K, np.zeros((3, 1))))
        O1 = np.zeros((3, 1), dtype=np.float32)

        # Camera 2 Projection Matrix K[R|t]
        P2 = np.hstack((R, t))
        P2 = K @ P2
        O2 = -R.T @ t

        nGood = 0

        for i, match in enumerate(vbMatchesInliers):

            kp1 = vKeys1[i]
            kp2 = vKeys2[i]

            # Triangulate points
            p3dC1 = self.triangulate(kp1, kp2, P1, P2)

            if not np.isfinite(p3dC1).all():
                vbGood[i] = False
                continue

            # Check parallax
            normal1 = p3dC1 - O1
            dist1 = np.linalg.norm(normal1)

            normal2 = p3dC1 - O2
            dist2 = np.linalg.norm(normal2)

            if dist1 <= 1e-9 or dist2 <= 1e-9:
                continue
            cosParallax = float(
                np.dot(normal1.flatten(), normal2.flatten()) / (dist1 * dist2)
            )
            cosParallax = float(np.clip(cosParallax, -1.0, 1.0))
    
            # Check depth in front of cameras
            # if depth in front of camera 1 < 1 or cos(parallax) < 0.99998, fails

            if p3dC1[2] <= 0:
                continue

            # if depth in front of camera 2 < 1 or cos(parallax) < 0.99998, fails

            p3dC2 = R @ p3dC1 + t

            if p3dC2[2] <= 0:
                continue

            # Check reprojection error in first image
            invZ1 = 1.0 / p3dC1[2]
            im1x = fx * p3dC1[0] * invZ1 + cx
            im1y = fy * p3dC1[1] * invZ1 + cy
            squareError1 = (im1x - kp1[0][0]) ** 2 + (im1y - kp1[0][1]) ** 2

            if squareError1 > th2:
                # if reprojection error in first image is larger than th2, fails
                continue

            # Check reprojection error in second image
            invZ2 = 1.0 / p3dC2[2]
            im2x = fx * p3dC2[0] * invZ2 + cx
            im2y = fy * p3dC2[1] * invZ2 + cy
            squareError2 = (im2x - kp2[0][0]) ** 2 + (im2y - kp2[0][1]) ** 2

            if squareError2 > th2:
                # if reprojection error in first image is larger than th2, fails
                continue

            vCosParallax.append(cosParallax)
            vP3D[i] = np.array([p3dC1[0], p3dC1[1], p3dC1[2]], dtype=np.float32)
            nGood += 1

            if cosParallax < self.min_point_parallax_cos:
                vbGood[i] = True

        good_indices = [index for index, good in enumerate(vbGood) if good]
        if good_indices:
            depths = np.array(
                [float(np.asarray(vP3D[index]).reshape(3)[2]) for index in good_indices]
            )
            median_depth = float(np.median(depths))
            max_depth = self.max_depth_factor * median_depth
            for index, depth in zip(good_indices, depths):
                if depth > max_depth:
                    vbGood[index] = False
                    vP3D[index] = None

        if nGood > 0:
            vCosParallax.sort()
            idx = min(self.settings.parallax_order_index, len(vCosParallax) - 1)
            parallax = np.degrees(np.arccos(np.clip(vCosParallax[idx], -1.0, 1.0)))
        else:
            parallax = 0

        return nGood, parallax, vP3D, vbGood


    def triangulate(self, kp1, kp2, P1, P2):
        """
        Triangulate a 3D point from two keypoints and their corresponding projection matrices.
        
        Args:
            kp1: KeyPoint from the first image (with .pt containing (x, y) coordinates).
            kp2: KeyPoint from the second image (with .pt containing (x, y) coordinates).
            P1: 3x4 projection matrix of the first camera.
            P2: 3x4 projection matrix of the second camera.
        
        Returns:
            x3D: A 3D point in Euclidean coordinates (numpy array of shape (3,)).
        """
        # Construct the matrix A for triangulation
        A = np.zeros((4, 4), dtype=np.float32)

        A[0, :] = kp1[0][0] * P1[2, :] - P1[0, :]
        A[1, :] = kp1[0][1] * P1[2, :] - P1[1, :]
        A[2, :] = kp2[0][0] * P2[2, :] - P2[0, :]
        A[3, :] = kp2[0][1] * P2[2, :] - P2[1, :]

        # Perform SVD decomposition
        _, _, Vt = cv2.SVDecomp(A, flags=cv2.SVD_MODIFY_A | cv2.SVD_FULL_UV)
        x3D_homogeneous = Vt[-1, :]  # Last row of Vt (or last column of V transposed)
        x3D_homogeneous /= x3D_homogeneous[3]  # Normalize to make it Euclidean (divide by w)

        return x3D_homogeneous[:3].reshape(-1,1)  # Return the 3D point (x, y, z)
    
    def warp_iou(self, H):
        """
        Warp img1 to img2 using the homography.

        Returns:
        - roi_in_new_image: List of 4 corners [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
 
        corners = np.array([[self.margin, self.margin], [self.width-self.margin, self.margin], [self.width-self.margin, self.height-self.margin], [self.margin, self.height-self.margin]], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1,2)
        warped_points = warped_corners.astype(int).tolist()
        return warped_points


    def compute_iou_from_homography_fast(self, H):
        """
        Warp img1 to img2 using the homography. Calculate the IoU between two rectangles defined by their corners.

        Parameters:
        - Homography matrix numpy.array 3x3

        Returns:
        - IoU: Intersection over Union value
        """
        # Warp img1 to img2 using the homography
        corners = np.array([[self.margin, self.margin], [self.width-self.margin, self.margin], [self.width-self.margin, self.height-self.margin], [self.margin, self.height-self.margin]], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).astype(int).reshape(-1,2)
        corners = corners.astype(int)
        # Extract bounding boxes from corners
        x1_min = min([pt[0] for pt in corners])
        y1_min = min([pt[1] for pt in corners])
        x1_max = max([pt[0] for pt in corners])
        y1_max = max([pt[1] for pt in corners])

        if x1_min < self.margin:
            x1_min = self.margin
        if y1_min < self.margin:
            y1_min = self.margin
        if x1_max > self.width-self.margin:
            x1_max = self.width-self.margin
        if y1_max > self.height-self.margin:
            y1_max = self.height-self.margin


        x2_min = min([pt[0] for pt in warped_corners])
        y2_min = min([pt[1] for pt in warped_corners])
        x2_max = max([pt[0] for pt in warped_corners])
        y2_max = max([pt[1] for pt in warped_corners])

        if x2_min < self.margin:
            x2_min = self.margin
        if y2_min < self.margin:
            y2_min = self.margin
        if x2_max > self.width-self.margin:
            x2_max = self.width-self.margin
        if y2_max > self.height-self.margin:
            y2_max = self.height-self.margin

        # Calculate intersection rectangle
        xi1 = max(x1_min, x2_min)
        yi1 = max(y1_min, y2_min)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)

        if xi1 < xi2 and yi1 < yi2:
            inter_area = (xi2 - xi1) * (yi2 - yi1)
        else:
            inter_area = 0

        # Compute areas of both rectangles
        rect1_area = (x1_max - x1_min) * (y1_max - y1_min)
        rect2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Compute union area
        union_area = rect1_area + rect2_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area
        return iou


    def compute_iou_from_rois(self, rect1, rect2):
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

        if x1_min < self.margin:
            x1_min = self.margin
        if y1_min < self.margin:
            y1_min = self.margin
        if x1_max > self.width-self.margin:
            x1_max = self.width-self.margin
        if y1_max > self.height-self.margin:
            y1_max = self.height-self.margin


        x1_1, y1_1, x2_1, y2_1 = (x1_min, y1_min, x1_max, y1_max)

        x2_min = min([pt[0] for pt in rect2])
        y2_min = min([pt[1] for pt in rect2])
        x2_max = max([pt[0] for pt in rect2])
        y2_max = max([pt[1] for pt in rect2])

        if x2_min < self.margin:
            x2_min = self.margin
        if y2_min < self.margin:
            y2_min = self.margin
        if x2_max > self.width-self.margin:
            x2_max = self.width-self.margin
        if y2_max > self.height-self.margin:
            y2_max = self.height-self.margin


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
            if x1_1 < xi1 - self.margin:
                non_overlap_coords_1.append([x1_min, y1_1, xi1, y2_1])  # Left part of rect1
            if x2_1 > xi2 + self.margin:
                non_overlap_coords_1.append([xi2, y1_1, x2_1, y2_1])  # Right part of rect1
            if y1_1 < yi1 - self.margin:
                non_overlap_coords_1.append([xi1, y1_1, xi2, yi1])  # Top part of rect1
            if y2_1 > yi2 + self.margin:
                non_overlap_coords_1.append([xi1, yi2, xi2, y2_1])  # Bottom part of rect1

            # Non-overlapping parts of rect2
            if x1_2 < xi1 - self.margin:
                non_overlap_coords_2.append([x1_2, y1_2, xi1, y2_2])  # Left part of rect2
            if x2_2 > xi2 + self.margin:
                non_overlap_coords_2.append([xi2, y1_2, x2_2, y2_2])  # Right part of rect2
            if y1_2 < yi1 - self.margin:
                non_overlap_coords_2.append([xi1, y1_2, xi2, yi1])  # Top part of rect2
            if y2_2 > yi2 + self.margin:
                non_overlap_coords_2.append([xi1, yi2, xi2, y2_2])  # Bottom part of rect2
        elif iou < 0.1:
            # If no intersection, the entire rectangles are non-overlapping
            non_overlap_coords_1.append(rect1)
            non_overlap_coords_2.append(rect2)
        return iou, intersection, non_overlap_coords_1, non_overlap_coords_2


    def check_features_visibility(self, undist_features):
        """
        Label points as True (inside) or False (outside) based on their position in the image.
        :return: A bool flag, A boolean array where True = inside, False = outside.
        """

        x1 = self.margin
        y1 = self.margin
        x2 = self.height-self.margin
        y2 = self.width-self.margin

        x = undist_features[:, 0]
        y = undist_features[:, 1]

        # Check bounds using NumPy
        inside_x = (x > x1) & (x < x2)
        inside_y = (y > y1) & (y < y2)
        inside = inside_x & inside_y  # Combine both conditions

        result = inside.reshape(-1, 1)
        # Return the result as a N, 1 array

        return result


