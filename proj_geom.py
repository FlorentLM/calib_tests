import cv2
import numpy as np


def extrinsics_mat(rvec, tvec, hom=False):
    """
        Convert rotation vector and translation vector to a 3x4 extrinsics matrix E

                                    [ r00, r01, r02 ]             [ t0 ]
            E = [R|t]      with R = [ r10, r11, r12 ]   and   t = [ t1 ]
                                    [ r20, r21, r22 ]             [ t2 ]
    """
    rvec = np.asarray(rvec).squeeze()
    tvec = np.asarray(tvec).squeeze()

    # Convert rotation vector into rotation matrix (and jacobian)
    R_mat, jacob = cv2.Rodrigues(rvec)
    # Insert R mat into the Transform matrix and append translation vector to last column
    E = np.hstack([R_mat, tvec[:, np.newaxis]])

    if hom:
        return np.vstack([E, np.array([0, 0, 0, 1])])
    else:
        return E


def to_rtvecs(extrinsics_mat):
    """
        Convert 3x4 (or 4x4) Extrinsics matrix to rotation vector and translation vector
    """
    rvec, jacob = cv2.Rodrigues(extrinsics_mat[:3, :3])
    tvec = extrinsics_mat[:3, 3]
    return rvec.squeeze(), tvec


def projection_mat(intrinsics_mat, extrinsics_mat):
    """
        Just a dot product of K o E to return the projection matrix P

           This matrix maps 3D points represented in real-world, camera-relative coordinates (X, Y, Z, 1)
           to 2D points in the image plane represented in normalized camera-relative coordinates (u, v, 1)

           2d_point     matrix_K           matrix_E         3d_point
                      (intrinsics)       (extrinsics)

                                                             [ X ]
             [ u ]   [ fx, 0, cx ]   [ r00, r01, r02, t0 ]   [ Y ]
             [ v ] = [ 0, fy, cy ] o [ r10, r11, r12, t1 ] o [ Z ]
             [ 1 ]   [ 0,  0,  1 ]   [ r20, r21, r22, t2 ]   [ 1 ]

                                KE = P
    """
    return np.dot(intrinsics_mat, extrinsics_mat)


def invert_extrinsics(rvec, tvec):
    rvec = np.asarray(rvec).squeeze()
    tvec = np.asarray(tvec).squeeze()

    R_mat, _ = cv2.Rodrigues(rvec)

    R_inv = np.linalg.inv(R_mat)  # or R_mat.T because the matrix is orthonormal
    tvec_inv = -R_inv @ tvec

    rvec_inv, _ = cv2.Rodrigues(R_inv)

    return rvec_inv.squeeze(), tvec_inv


def invert_extrinsics_2(rvec, tvec):
    rt_mat = extrinsics_mat(rvec, tvec, hom=True)
    rt_inv = np.linalg.inv(rt_mat)
    return to_rtvecs(rt_inv)


def invert_extrinsics_mat(extrinsics_mat):

    if extrinsics_mat.shape[0] == 3:
        extrinsics_mat = np.vstack([extrinsics_mat, np.array([0, 0, 0, 1])])
        return np.linalg.inv(extrinsics_mat)[:3, :]
    else:
        return np.linalg.inv(extrinsics_mat)


def back_projection(points2d, depth, intrinsics_mat, extrinsics_mat, invert=True):
    """
    Performs back-projection from 2D image coordinates to 3D world coordinates.

        Parameters
        ----------
        points2d : 2D image coordinates X, Y
        depth : The depth value (Z coordinate) at the given 2D image points
        intrinsics_mat : The intrinsics camera matrix K
        extrinsics_mat : The extrinsics camera matrix [R|t]
        invert : Whether or not to invert the extrinsics (True: projects to camera-centric coordinates
                                                            False: projects to object-centric coordinates)

        Returns: Array of the 3D world coordinates for given depth

    """
    # TODO - Add undistortion using dist_coeffs

    if not (isinstance(depth, int) or isinstance(depth, float)) and np.atleast_1d(depth).shape[0] != points2d.shape[0]:
        raise AssertionError('Depth vector length does not match 2D points array')

    # 2D image coordinates -> normalized camera coordinates
    if points2d.ndim == 1:
        homogeneous_2dcoords = np.array([*points2d, 1])
    else:
        homogeneous_2dcoords = np.c_[points2d, np.ones(points2d.shape[0])]

    normalised_coords = np.linalg.inv(intrinsics_mat) @ homogeneous_2dcoords.T

    # Depth
    normalised_coords *= depth

    if invert:
        extrinsics_mat = invert_extrinsics_mat(extrinsics_mat)

    R = extrinsics_mat[:3, :3]
    tvec = extrinsics_mat[:3, 3]

    if points2d.ndim == 1:
        # Convert normalized camera coordinates to world coordinates
        points3d = R @ normalised_coords + tvec
    else:
        points3d = (R @ normalised_coords + tvec[:, np.newaxis]).T

    return points3d


def triangulate_points(points2d, projection_matrices):
    """
    Triangulate 3D point from multiple 2D points and their corresponding camera matrices

        For each i-th 2D point and its corresponding camera matrix, two rows are added to matrix A:

                [ u_1 * P_1_3 - P_1_1 ]
                [ v_1 * P_1_3 - P_1_2 ]
         A =    [ u_2 * P_2_3 - P_2_1 ]
                [ v_2 * P_2_3 - P_2_2 ]
                [          ...        ]
                [          ...        ]
                [          ...        ]

       where P_i_j denotes the j-th row of the i-th camera matrix

    We use SVD to solve the system AX=0. The solution X is the last row of V^t from SVD
    We then normalize the solution to represent a 3D point in homogeneous coordinates

    See https://people.math.wisc.edu/~chr/am205/g_act/svd_slides.pdf for more info and sources

    Parameters
    ----------
    points2d:     List of n 2D points from different cameras, each as (u, v)
    projection_matrices: List of n projection matrices

    Returns: Array of n 3D points coordinates

    """

    nb_views = len(points2d)
    if nb_views != len(projection_matrices):
        raise ValueError("Number of 2D points series must match the number of projection matrices!")

    A = np.zeros(nb_views * 2)
    for i in range(0, nb_views, 2):
        P = projection_matrices[i]
        u, v = points2d[i]
        A[i] = u * P[2] - P[0]
        A[i+1] = v * P[2] - P[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]  # Ensure homogeneous coordinates

    return X[:3]


def perspective_function(pixel_length, camera_matrix, tvec):
    """
        Calculates the real_world length from a pixel length
    """
    tvec = np.asarray(tvec).squeeze()
    f = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0
    return (pixel_length * tvec[2] / f).squeeze()

