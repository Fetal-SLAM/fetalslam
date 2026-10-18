"""A continuous rectangular grid fitted to robust planar observations."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def build_plane_mesh(points, settings, check_cancelled=lambda: None):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = np.unique(points[np.isfinite(points).all(axis=1)], axis=0)
    if len(points) < settings['min_sparse_points']:
        raise ValueError('Too few points for plane fitting')
    check_cancelled()
    # Robust extent avoids letting distant outliers set the RANSAC tolerance.
    extent = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
    scale = np.linalg.norm(extent)
    if scale <= 1e-12:
        raise ValueError('Degenerate plane extent')
    singular = np.linalg.svd(points - np.median(points, axis=0), compute_uv=False)
    if singular[1] <= max(singular[0] * settings['collinear_ratio'], 1e-12):
        raise ValueError('Plane observations are collinear')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, indices = cloud.segment_plane(
        distance_threshold=scale * settings['plane_distance_ratio'],
        ransac_n=3, num_iterations=settings['plane_ransac_iterations'])
    check_cancelled()
    if len(indices) < max(settings['min_sparse_points'],
                          int(np.ceil(len(points) * settings['plane_min_inlier_ratio']))):
        raise ValueError('Insufficient support for a dominant plane')
    inliers = points[indices]
    origin = inliers.mean(axis=0)
    _, singular, axes = np.linalg.svd(inliers - origin, full_matrices=False)
    if singular[1] <= max(singular[0] * settings['collinear_ratio'], 1e-12):
        raise ValueError('Plane observations are collinear')
    # Keep original signed heights; projecting them away would erase relief.
    chart = (inliers - origin) @ axes[:2].T
    heights = (inliers - origin) @ axes[2]
    lower, upper = chart.min(axis=0), chart.max(axis=0)
    size = upper - lower
    cells = np.maximum(1, np.ceil(size / size.max() * settings['plane_grid_cells']).astype(int))
    nx, ny = cells + 1
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], nx),
                         np.linspace(lower[1], upper[1], ny))
    uv = np.column_stack((xx.ravel(), yy.ravel()))
    unique_chart, keep = np.unique(chart, axis=0, return_index=True)
    tree = cKDTree(unique_chart)
    spacing = np.median(tree.query(unique_chart, k=2)[0][:, 1])
    radius = settings['plane_support_radius_factor'] * spacing
    distances, neighbors = tree.query(uv, k=min(settings['plane_neighbors'], len(unique_chart)))
    if distances.ndim == 1:
        distances, neighbors = distances[:, None], neighbors[:, None]
    valid = distances <= radius
    weights = np.exp(-0.5 * (distances / max(radius, 1e-12)) ** 2) * valid
    weight_sum = weights.sum(axis=1)
    supported = weight_sum > 0
    target = np.zeros(len(uv))
    target[supported] = ((weights * heights[keep[neighbors]]).sum(axis=1)[supported]
                         / weight_sum[supported])
    target = target.reshape(ny, nx)
    supported = supported.reshape(ny, nx)
    height = target.copy()
    # Only heights move: footprint and topology cannot shrink. Unsupported
    # vertices remain on the plane, rather than extrapolating distant relief.
    for _ in range(settings['plane_smooth_iterations']):
        check_cancelled()
        total, degree = np.zeros_like(height), np.zeros_like(height)
        total[1:] += height[:-1]
        total[:-1] += height[1:]
        total[:, 1:] += height[:, :-1]
        total[:, :-1] += height[:, 1:]
        degree[1:] += 1
        degree[:-1] += 1
        degree[:, 1:] += 1
        degree[:, :-1] += 1
        data_weight = settings['plane_data_weight']
        smooth_weight = settings['plane_smooth_weight']
        height = (data_weight * target + smooth_weight * total) / (data_weight + smooth_weight * degree)
        height[~supported] = 0
    vertices = origin + uv @ axes[:2] + height.ravel()[:, None] * axes[2]
    index = np.arange(nx * ny).reshape(ny, nx)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    triangles = np.vstack((np.column_stack((a, b, d)), np.column_stack((a, d, c))))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                   o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    return mesh, {'inliers': len(inliers), 'input_points': len(points),
                  'grid_shape': (int(ny), int(nx)), 'supported_vertices': int(supported.sum()),
                  'corners': origin + np.array([lower, [upper[0], lower[1]], upper,
                                                [lower[0], upper[1]]]) @ axes[:2]}
