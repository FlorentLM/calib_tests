from collections import deque
from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import toml

import proj_geom
import utilities

FOLDER = Path(f'D:\\MokapRecordings\\persie-240716\\calib')

captures = {}

vid_lengths = []
for f in FOLDER.glob('*.mp4'):
    shape, nb_frames = utilities.probe_video(f)
    captures[f.stem] = cv2.VideoCapture(f.as_posix())
    vid_lengths.append(nb_frames)

nb_frames = min(vid_lengths)

frame_stack = np.hstack([np.zeros(shape, dtype=np.uint8) for _ in range(len(captures.keys()))])

##

# w_name = 'detection'
# cv2.namedWindow(w_name)
#
# def on_slider_change(trackbar_value):
#
#     for i, cap in enumerate(captures):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
#         r, frame = cap.read()
#         if r:
#             w = frame.shape[1]
#             frame_stack[:, i * w:i * w + w] = frame
#
#     img = cv2.resize(frame_stack, (frame_stack.shape[1] // 3, frame_stack.shape[0] // 3))
#     cv2.imshow(w_name, img)
#     pass
#
# cv2.createTrackbar('Frame', w_name, 0, nb_frames, on_slider_change)
# on_slider_change(0)
#
# while True:
#     k = cv2.waitKeyEx(1)
#     #  ---Windows--    ---macOS--
#     if k == 2424832 or k == 63234:
#         p = cv2.getTrackbarPos('Frame', w_name)
#         cv2.setTrackbarPos('Frame', w_name, pos=max(0, p-1))
#     elif k == 2555904 or k == 63235:
#         p = cv2.getTrackbarPos('Frame', w_name)
#         cv2.setTrackbarPos('Frame', w_name, pos=min(nb_frames, p+1))
#     elif k == 27:
#         break
#
# cv2.destroyAllWindows()
#
# ##

from intrinsics import IntrinsicsTool

# Board parameters to detect
BOARD_COLS = 5                      # Total rows in the board (chessboard)
BOARD_ROWS = 6                     # Total cols in the board
SQUARE_LENGTH_MM = 1.5                # Length of one chessboard square in real life units (i.e. mm)
MARKER_BITS = 4                     # Size of the markers in 'pixels' (not really, but you get the idea)

charuco_board = utilities.generate_charuco(board_rows=BOARD_ROWS,
                                           board_cols=BOARD_COLS,
                                           square_length_mm=SQUARE_LENGTH_MM,
                                           marker_bits=MARKER_BITS)

calib = IntrinsicsTool(charuco_board)

fr_id = 987

rvecs = []
tvecs = []
for name, cap in captures.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr_id)
    r, frame = cap.read()
    if r:
        calib.load(FOLDER / f'{name}.toml')
        calib.load_frame(frame)

        calib.detect_markers(refine=True)
        calib.detect_corners(refine=True)
        calib.reproject()

        rvecs.append(calib.rvec)
        tvecs.append(calib.tvec)
        # test = calib.visualise()

def inv_camera_position(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.transpose(R)
    tvec_inv = -np.dot(R_inv, tvec)
    return R_inv, tvec_inv

##

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(5):

    R_inv, tvec_inv = inv_camera_position(rvecs[i], tvecs[i])

    ax.scatter(tvec_inv[0], tvec_inv[1], tvec_inv[2], label=f'Camera {i + 1}')

    camera_origin = tvec_inv
    x_axis = camera_origin + R_inv[:, 0]
    y_axis = camera_origin + R_inv[:, 1]
    z_axis = camera_origin + R_inv[:, 2]

    ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
              x_axis[0] - camera_origin[0], x_axis[1] - camera_origin[1], x_axis[2] - camera_origin[2], color='r')
    ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
              y_axis[0] - camera_origin[0], y_axis[1] - camera_origin[1], y_axis[2] - camera_origin[2], color='g')
    ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
              z_axis[0] - camera_origin[0], z_axis[1] - camera_origin[1], z_axis[2] - camera_origin[2], color='b')

for corner in calib.board_corners_3d:
    ax.scatter(corner[0], corner[1], corner[2], color='black', marker='x')

ax.legend()
plt.show()