import numpy as np
import cv2
from collections import deque
from pathlib import Path
np.set_printoptions(precision=3, suppress=True, threshold=5)
import utilities


# This script generates the following:
# ===================================
#
#
# tvecs: Translation vectors of 3, in mm (or whatever real-world unit the Charuco board is in).
# -----     They represent the position of the camera in model coordinates, for each of the frames
#           where the board is detected (i.e. the origin is a point on the Charuco board)
#
# rvecs: Rotation vectors of 3, in radians.
# -----     They represent the orientation of the camera in model coordinates, for each of the frames
#           where the board is detected (i.e. the origin is a point on the Charuco board)
#
# camera_matrix: a 3x3 matrix of the camera intrinsics parameters
# -------------
#
#       [ fx, 0, cx ]
#   K = [ 0, fy, cy ]
#       [ 0,  0,  1 ]
#
#       fx and fy are the focal lengths along the x and y axes
#           The units are pixels per unit length in the scene, but since they are derived from real-world focal lengths
#           (e.g. millimeters) scaled by the pixel size (e.g. millimeters per pixel), their effective unit is pixels.
#
#       cx and cy are the coordinates of the principal point (in pixels)
#           This corresponds to the point where the optical axis intersects the image plane
#           and is usually in the centre of the frame.
#
# dist_coeffs: a vector of 5 (or 8 if using a more complex model than a pinhole camera)
# -----------
#
# D = [ k1, k2, p1, p2, k3 ]
#
#       k1, k2, k3 (and eventually k4, k5, k6) are coefficients that describe radial distortion.
#           Radial distortion causes straight lines to appear curved, the wider angle the camera, the bigger the effect.
#
#       p1, p2 are coefficients that describe tangential distortion.
#           Tangential distortion occurs when the lens and the image plane are not perfectly parallel.
#
#       The coefficients are dimensionless, they are used in polynomial equations to adjust the pixel coordinates.
#
# ==============================================================
# More info on https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html


# Board parameters to detect
BOARD_COLS = 5                      # Total rows in the board (chessboard)
BOARD_ROWS = 6                     # Total cols in the board
SQUARE_LENGTH_MM = 1.5                # Length of one chessboard square in real life units (i.e. mm)
MARKER_BITS = 4                     # Size of the markers in 'pixels' (not really, but you get the idea)

# Video to load
FOLDER = Path(f'D:\\MokapRecordings\\persie-240716\\calib')
FILE = 'cam0_avocado_session32.mp4'

SAVE = False                        # Whether to save the calibration or no
REPROJ_ERR = 0.2                    # Reprojection error we deem acceptable

##

# Generate Charuco board and corresponding detector
aruco_dict, charuco_board = utilities.generate_charuco(BOARD_ROWS, BOARD_COLS,
                                             square_length_mm=SQUARE_LENGTH_MM,
                                             marker_bits=MARKER_BITS)

# Optionally save the board as a svg for printing
# utilities.print_board(charuco_board, multi_size=False)
# utilities.print_board(charuco_board, multi_size=True, factor=1.25)

# Create a detector (default parameters) and define stop criteria for board corners refinement
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

nb_total_markers = len(charuco_board.getIds())
nb_total_corners = len(charuco_board.getChessboardCorners())

##

video_path = FOLDER / FILE
shape, nb_frames = utilities.probe_video(video_path)

cap = cv2.VideoCapture(video_path.as_posix())

w_name = 'detection'

# Init some global variables
coverage = np.full(shape[:2], False, dtype=bool)    # This is the area of the image that has been covered so far

# These will store detection data for each frame
all_frames_corners = deque()
all_frames_ids = deque()
all_frames_frame_ids = deque()

# These are the ones we want to compute, we initialise them to None for the first estimation without a prior
camera_matrix = None
dist_coeffs = None


# This is the function that does the bulk of the work
def detect(frame, frame_id=None):
    nb_detected_markers = 0
    nb_detected_squares = 0

    if frame.ndim == 3:
        frame_mono = frame[:, :, 0]
        frame_col = frame
    else:
        frame_mono = frame
        frame_col = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Detect and refine aruco markers
    marker_corners, marker_ids, rejected = detector.detectMarkers(frame_mono)
    marker_corners, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
        image=frame_mono,
        board=charuco_board,
        detectedCorners=marker_corners,
        detectedIds=marker_ids,
        rejectedCorners=rejected,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs)

    if marker_ids is not None:
        nb_detected_markers = len(marker_ids)

    # If any marker has been detected, try to detect the board corners
    if nb_detected_markers > 0:

        nb_detected_squares, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=frame_mono,
            board=charuco_board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            minMarkers=1)
        try:
            # Refine the board corners
            charuco_corners = cv2.cornerSubPix(frame_mono, charuco_corners,
                                               winSize=(20, 20),
                                               zeroZone=(-1, -1),
                                               criteria=criteria)
        except:
            pass

        # If corners have been found, show them as red dots
        if charuco_corners is not None:
            for xy in charuco_corners[:, 0]:
                frame_col = cv2.circle(frame_col, np.round(xy).astype(int), 2, (0, 0, 255), 2)

            # Compute image area with detection
            markers_bin_ctr = utilities.markers_binary_contour(frame, marker_corners)
            hull_pts = utilities.hull_coords(markers_bin_ctr)
            hull_bin_ctr = utilities.binary_contour(frame_mono, hull_pts)
            detected_area = utilities.fill_binary_contour(hull_bin_ctr)

            # Newly detected area is the union of the current detection and the inverse of the overlap with existing
            overlap = np.logical_and(detected_area, coverage)
            new_area = np.logical_and(detected_area, ~overlap)

            # If the new frame brings sufficiently new coverage, add its data to the list
            if new_area.mean() * 100 >= 0.2 and len(charuco_corners) > 5:
                all_frames_corners.append(charuco_corners)
                all_frames_ids.append(charuco_ids)
                coverage[detected_area] = True
                if frame_id is not None:
                    all_frames_frame_ids.append(frame_id)

    # Coloured overlay of the coverage
    coverage_overlay = np.zeros_like(frame_col)
    coverage_overlay[coverage, 1] = 255     # channel 1 = green

    # Add the overlay to the visualisation image
    frame_col = cv2.addWeighted(frame_col, 0.85, coverage_overlay, 0.15, 0)

    # Add information text to the visualisation image
    frame_col = cv2.putText(frame_col, f"Aruco markers: {nb_detected_markers}/{nb_total_markers}", (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    frame_col = cv2.putText(frame_col, f"Corners: {nb_detected_squares}/{nb_total_corners}", (30, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    frame_col = cv2.putText(frame_col, f"Area: {coverage.mean() * 100:.2f}% ({len(all_frames_corners)} snapshots)", (30, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Show
    cv2.imshow(w_name, frame_col)


##

# Run mini GUI
def on_slider_change(trackbar_value):
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
    r, frame = cap.read()
    if r:
        detect(frame, frame_id=trackbar_value)
    pass

cv2.namedWindow(w_name)
cv2.createTrackbar('Frame', w_name, 0, nb_frames, on_slider_change)
on_slider_change(0)

while True:
    k = cv2.waitKeyEx(1)
    #  ---Windows--    ---macOS--
    if k == 2424832 or k == 63234:
        p = cv2.getTrackbarPos('Frame', w_name)
        cv2.setTrackbarPos('Frame', w_name, pos=max(0, p-1))
    elif k == 2555904 or k == 63235:
        p = cv2.getTrackbarPos('Frame', w_name)
        cv2.setTrackbarPos('Frame', w_name, pos=min(nb_frames, p+1))
    elif k == 27:
        break

cv2.destroyAllWindows()

##

# TODO - Integrate the functions below to the detection loop for iterative refinement

# Compute calibration using all the frames we selected
retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_frames_corners,
    charucoIds=all_frames_ids,
    board=charuco_board,
    imageSize=shape[:2],
    cameraMatrix=camera_matrix,
    distCoeffs=dist_coeffs,
    flags=cv2.CALIB_USE_QR)

# Reproject known 3D points (the board corners) using the freshly computed calibration to estimate the reprojection error
objpoints = [charuco_board.getChessboardCorners()[ids] for ids in all_frames_ids]
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(all_frames_corners[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
    print(f"Frame {all_frames_frame_ids[i]} error: {error:.2f} px")
avg_err = mean_error / len(objpoints)
print(f"Average error: {avg_err:.2f} px")


# Save calibration data to disk if the reprojection error is ok
if SAVE:
    cam_name = FILE.split('.')[0]
    if avg_err < REPROJ_ERR:
        print(f"Calibration successful! Saving.")
        np.savez_compressed(FOLDER / f'{cam_name}_frames_ids.npz', np.array(list(all_frames_frame_ids)))
        np.savez_compressed(FOLDER / f'{cam_name}_camera_matrix.npz', camera_matrix)
        np.savez_compressed(FOLDER / f'{cam_name}_dist_coeffs.npz', dist_coeffs)
        np.savez_compressed(FOLDER / f'{cam_name}_rvecs.npz', np.hstack(rvecs).T)
        np.savez_compressed(FOLDER / f'{cam_name}_tvecs.npz', np.hstack(tvecs).T)
    else:
        print(f"Calibration is meh... Not saving")
