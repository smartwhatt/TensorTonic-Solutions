import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.asarray(points)
    if len(points.shape) == 1:
        points = points.reshape(points.shape[0], 1)
    else:
        points = points.T

    points = np.vstack((points, np.ones(points.shape[-1])))

    points_new = T@points
    result = points_new[:3, :].T

    if result.shape[0] == 1:
        result = result.reshape(result.shape[-1])

    return result