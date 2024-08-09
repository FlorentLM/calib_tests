import numpy as np
from scipy.optimize import least_squares
import utilities
from intrinsics import CalibrationTool
import cv2
from pathlib import Path


def flatten_extrinsics(rvecs, tvecs):
    return np.hstack([np.array(rvecs), np.array(tvecs)]).ravel()

def unflatten_extrinsics(params):
    r, t = np.split(params.reshape(-1, 6), [3], axis=1)
    return r, t

def flatten_intrinsics(camera_matrices, distortion_coeffs):
    return np.hstack([np.array(camera_matrices).reshape(-1, 9)[:, [0, 2, 4, 5]], np.array(distortion_coeffs)]).ravel()

def unflatten_intrinsics(params):
    c, d = np.split(params.reshape(-1, 9), [4], axis=1)
    cm = np.zeros((d.shape[0], 9))
    cm[:, [0, 2, 4, 5]] = c
    cm[:, -1] = 1
    return cm.reshape(-1, 3, 3), d

def flatten_ext_int(rvecs, tvecs, camera_matrices, distortion_coeffs):
    r = np.array(rvecs)
    t = np.array(tvecs)
    c = np.array(camera_matrices).reshape(-1, 9)[:, [0, 2, 4, 5]]
    d = np.array(distortion_coeffs)
    return np.hstack([r, t, c, d]).ravel()

def weights_ext_int(nb_observations, rvecs_w=1.0, tvecs_w=1.0, camera_matrices_w=1.0, distortion_coeffs_w=1.0):
    r = np.ones((nb_observations, 3)) * rvecs_w
    t = np.ones((nb_observations, 3)) * tvecs_w
    c = np.ones((nb_observations, 4)) * camera_matrices_w
    d = np.ones((nb_observations, 5)) * distortion_coeffs_w
    return np.hstack([r, t, c, d]).ravel()

def unflatten_ext_int(params):
    r, t, c, d = np.split(params.reshape(-1, 15), [3, 6, 10], axis=1)
    cm = np.zeros((r.shape[0], 9))
    cm[:, [0, 2, 4, 5]] = c
    cm[:, -1] = 1
    return r, t, cm.reshape(-1, 3, 3), d

##

def cost_func(params, camera_matrices, distortion_coeffs, points_2d_observed, points_3d_observed):
    nb_observations = len(points_2d_observed)
    rvecs, tvecs = unflatten_extrinsics(params)

    error = []
    for i in range(nb_observations):
        reprojected_points, _ = cv2.projectPoints(points_3d_observed[i], rvecs[i], tvecs[i], camera_matrices[i], distortion_coeffs[i])
        error.append((points_2d_observed[i] - reprojected_points.squeeze().ravel()))
    return np.concatenate(error)


def cost_func_2(params, points_2d_observed, points_3d_observed):
    nb_observations = len(points_2d_observed)
    rvecs, tvecs, camera_matrices, distortion_coeffs = unflatten_ext_int(params)

    error = []
    for i in range(nb_observations):
        reprojected_points, _ = cv2.projectPoints(points_3d_observed[i], rvecs[i], tvecs[i], camera_matrices[i], distortion_coeffs[i])
        error.append((points_2d_observed[i] - reprojected_points.squeeze().ravel()))
    return np.concatenate(error)


def cost_func_3(params, x0, points_2d_observed, points_3d_observed, weights=None):
    nb_observations = len(points_2d_observed)
    rvecs, tvecs, camera_matrices, distortion_coeffs = unflatten_ext_int(params)

    error = []
    for i in range(nb_observations):
        reprojected_points, _ = cv2.projectPoints(points_3d_observed[i],
                                                  rvecs[i], tvecs[i],
                                                  camera_matrices[i],
                                                  distortion_coeffs[i])
        error.append((points_2d_observed[i] - reprojected_points.squeeze().ravel()))

    if weights is not None:
        # weighted difference from initial parameters
        param_diff = (params - x0) * weights
        error = np.concatenate([np.concatenate(error), param_diff])
    else:
        error = np.concatenate(error)

    return error

##


FOLDER = Path('/Users/florent/Desktop/cajal_messor_videos/calibration')

captures = {}

vid_lengths = []
for f in FOLDER.glob('*.mp4'):
    shape, nb_frames = utilities.probe_video(f)
    captures[f.stem] = cv2.VideoCapture(f.as_posix())
    vid_lengths.append(nb_frames)

##

BOARD_COLS = 7                      # Total rows in the board (chessboard)
BOARD_ROWS = 10                     # Total cols in the board
SQUARE_LENGTH_MM = 5                # Length of one chessboard square in real life units (i.e. mm)
MARKER_BITS = 4                     # Size of the markers in 'pixels' (not really, but you get the idea)

charuco_board = utilities.generate_charuco(board_rows=BOARD_ROWS,
                                           board_cols=BOARD_COLS,
                                           square_length_mm=SQUARE_LENGTH_MM,
                                           marker_bits=MARKER_BITS)
calib = CalibrationTool(charuco_board)

##

if __name__ == "__main__":

    nb_cams = 3
    frames = [162, 636, 1069, 3732, 4070]
    nb_frames = len(frames)

    # These are of unknown length, so lists and not arrays
    rvecs_l = []
    tvecs_l = []
    camera_matrices_l = []
    distortion_coeffs_l = []

    points_2d_observed_l = []
    points_3d_observed_l = []
    points_ids_l = []

    observed_points_l = []

    for j, fr in enumerate(frames):
        print(f"Frame {fr + 1}/{nb_frames}:")

        cams_l = []
        for i, (name, cap) in enumerate(captures.items()):
            calib.load(FOLDER / f'{name}.toml')
            print(f"  Camera {name}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)

            r, frame = cap.read()
            if r:
                calib.detect(frame)
                calib.reproject()

                if calib.has_intrinsics and calib.has_detection and calib.has_extrinsics:
                    rvecs_l.append(calib.rvec)
                    tvecs_l.append(calib.tvec)
                    camera_matrices_l.append(calib.camera_matrix)
                    distortion_coeffs_l.append(calib.dist_coeffs)

                    points_2d_observed_l.append(calib.points2d_detect.ravel())
                    points_3d_observed_l.append(calib.points3d[calib.ids_detect.ravel()])

                    points_ids_l.append(calib.ids_detect.ravel())

                    cams_l.append(name)

        observed_points_l.append({fr: cams_l})

        # Initial parameters
        x0 = flatten_ext_int(rvecs_l, tvecs_l, camera_matrices_l, distortion_coeffs_l)

        # weights = weights_ext_int(nb_observations=len(rvecs_l), rvecs_w=2.0, tvecs_w=2.0, camera_matrices_w=0.5, distortion_coeffs_w=0.5)

        result = least_squares(cost_func_2, x0, verbose=2, x_scale='jac', ftol=1e-8, method='trf', args=(points_2d_observed_l, points_3d_observed_l))

        # Extract optimized parameters
        optimized_params = result.x
        rvecs_opt, tvecs_opt, camera_matrices_opt, distortion_coeffs_opt = unflatten_ext_int(optimized_params)

        nb_observations = rvecs_opt.shape[0]
        error = []
        for i in range(nb_observations):
            reprojected_points, _ = cv2.projectPoints(points_3d_observed_l[i], rvecs_opt[i], tvecs_opt[i], camera_matrices_opt[i], distortion_coeffs_opt[i])
            error.append((points_2d_observed_l[i] - reprojected_points.squeeze().ravel()))
        error = np.concatenate(error)
        print(f'Mean error: {np.abs(error).mean()}')


    # Convert back to dict format (same as in the extrinsics_test. py file) just to test something
    rvecs = {}
    tvecs = {}
    camera_matrices = {}
    distortion_coeffs = {}

    i = 0
    for fr in observed_points_l:
        frame_nb = next(k for k in fr.keys())

        rvecs[frame_nb] = {}
        tvecs[frame_nb] = {}
        camera_matrices[frame_nb] = {}
        distortion_coeffs[frame_nb] = {}

        for name in fr[frame_nb]:
            rvecs[frame_nb][name] = rvecs_opt[i]
            tvecs[frame_nb][name] = tvecs_opt[i]
            camera_matrices[frame_nb][name] = camera_matrices_opt[i]
            distortion_coeffs[frame_nb][name] = distortion_coeffs_opt[i]

            i += 1
