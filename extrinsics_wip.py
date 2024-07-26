from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import proj_geom
import utilities
from intrinsics import CalibrationTool


FOLDER = Path('/Users/florent/Desktop/cajal_messor_videos/calibration')

captures = {}

vid_lengths = []
for f in FOLDER.glob('*.mp4'):
    shape, nb_frames = utilities.probe_video(f)
    captures[f.stem] = cv2.VideoCapture(f.as_posix())
    vid_lengths.append(nb_frames)

frame_stack = np.hstack([np.zeros(shape, dtype=np.uint8) for _ in range(len(captures.keys()))])

##

BOARD_COLS = 7                      # Total rows in the board (chessboard)
BOARD_ROWS = 10                     # Total cols in the board
SQUARE_LENGTH_MM = 5                # Length of one chessboard square in real life units (i.e. mm)
MARKER_BITS = 4                     # Size of the markers in 'pixels' (not really, but you get the idea)


charuco_board = utilities.generate_charuco(board_rows=BOARD_ROWS,
                                           board_cols=BOARD_COLS,
                                           square_length_mm=SQUARE_LENGTH_MM,
                                           marker_bits=MARKER_BITS)

##

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

calib = CalibrationTool(charuco_board)

rvecs = {}
tvecs = {}
camera_matrices = {}
distortion_coeffs = {}

h, w = shape[:2]
points2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

recentre = False

camera_colors = ['red', 'green', 'blue']
# camera_colors = ['orange', 'pink', 'purple']

frames = [162, 636, 1069, 3732, 4070]

for j, fr in enumerate(frames):
    origin = None

    rvecs[fr] = {}
    tvecs[fr] = {}
    camera_matrices[fr] = {}
    distortion_coeffs[fr] = {}

    for i, (name, cap) in enumerate(captures.items()):
        cam_color = camera_colors[i]

        calib.load(FOLDER / f'{name}.toml')

        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        r, frame = cap.read()
        if r:
            calib.detect(frame)
            calib.reproject()

            # if name in rvecs[fr].keys():
            if calib.has_extrinsics:
                rvecs[fr][name] = calib.rvec
                tvecs[fr][name] = calib.tvec
                camera_matrices[fr][name] = calib.camera_matrix
                distortion_coeffs[fr][name] = calib.dist_coeffs

                if recentre:
                    camera_matrices[fr][name][: 2, 2] = shape[1] / 2.0, shape[0] / 2.0

                # Extrinsics matrix in object-space
                ext_mat = proj_geom.extrinsics_mat(rvecs[fr][name], tvecs[fr][name], hom=True)
                if origin is None:
                    origin = ext_mat
                inv_mat = np.linalg.inv(ext_mat)

                transform_origin_mat = np.dot(origin, inv_mat)

                # The T part of the inverse of the transformation matrix is the camera's position
                cam_point3d = transform_origin_mat[:3, 3]

                # Plot cameras as dots
                ax.scatter(*cam_point3d, label=f'{name} ({fr})', color=cam_color, alpha=1.0)

                # # Plot frustum to far point = depth
                frustum_points3d = proj_geom.back_projection(points2d, 130, camera_matrices[fr][name], transform_origin_mat, invert=False)

                ax.add_collection3d(Poly3DCollection([frustum_points3d], facecolors=cam_color, linewidths=2, alpha=0.05))
                for corner in frustum_points3d:
                    ax.plot([cam_point3d[0], corner[0]], [cam_point3d[1], corner[1]], [cam_point3d[2], corner[2]], color=cam_color, linestyle=':', linewidth=0.5, alpha=0.5)

    # Add Charuco board
    board_points3d = calib.corners3d - inv_mat[:3, 3]
    board_points3d = np.dot(board_points3d, inv_mat[:3, :3])

    ax.add_collection3d(Poly3DCollection([board_points3d], facecolors='k', linewidths=2, alpha=0.05))
    ax.scatter(*board_points3d[0], color='r', marker='x')

ax.set_aspect('equal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_axis_off()
ax.legend()

plt.show()
