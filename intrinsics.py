from collections import deque
from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2
import toml

import proj_geom
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
FILE = 'cam4_blueberry_session32.mp4'

SAVE = False                        # Whether to save the calibration or no
REPROJ_ERR = 2.0                    # Reprojection error we deem acceptable (in pixels)

charuco_board = utilities.generate_charuco(board_rows=BOARD_ROWS,
                                           board_cols=BOARD_COLS,
                                           square_length_mm=SQUARE_LENGTH_MM,
                                           marker_bits=MARKER_BITS)

# Optionally save the board as a svg for printing
# utilities.print_board(charuco_board, multi_size=False)
# utilities.print_board(charuco_board, multi_size=True, factor=1.25)

## -------------------------------------------------------------

class IntrinsicsTool:

    def __init__(self, charuco_board, min_samples=25, max_samples=100):

        # The image on which to detect
        self.frame_in = None

        # Charuco board and detector parameters
        self.board = charuco_board
        aruco_dict = self.board.getDictionary()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        # Maximum number of markers and board points
        self.total_markers = len(self.board.getIds())
        self.total_points = len(self.board.getChessboardCorners())

        # Default attributes for markers and points coordinates and IDs
        self.markers_coords = np.array([])
        self.marker_ids = np.array([])
        self.nb_markers = 0

        self.points_coords = np.array([])
        self.points_ids = np.array([])
        self.nb_points = 0

        self.reprojected_points = np.array([])
        self.nb_reprojected = 0

        self.reprojected_corners = np.array([])

        # Default attributes for camera pose (in board-centric coordinates)
        self.rvec = np.array([])
        self.tvec = np.array([])

        # Create 3D coordinates for board corners (in board-centric coordinates)
        board_cols, board_rows = self.board.getChessboardSize()
        self.board_corners_3d = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]], dtype=float) * [board_cols, board_rows, 0] * self.board.getSquareLength()

        # This will be the area of the image that has been covered so far
        self.coverage = None

        # This will be the visualisation overlay
        self.coverage_overlay = None

        # These two deque() will store detected points for several samples
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.multi_samples_points_coords = deque(maxlen=self.max_samples)
        self.multi_samples_points_ids = deque(maxlen=self.max_samples)

        # These are the intrinsics we want to compute, we initialise them to None for 1st estimation without a prior
        self.camera_matrix = None
        self.dist_coeffs = None

        # Initialise reprojection errors to +inf
        self.best_error_px = float('inf')
        self.curr_error_px = float('inf')
        self.curr_error_mm = float('inf')

    @property
    def objpoints(self):
        """ Returns the coordinates of the chessboard corners in 3D, board-centric coordinates """
        return charuco_board.getChessboardCorners()

    def detect_markers(self, refine=True):

        self.markers_coords = np.array([])
        self.marker_ids = np.array([])
        self.nb_markers = 0

        # Detect and refine aruco markers
        marker_corners, marker_ids, rejected = self.detector.detectMarkers(self.frame_in)
        if refine:
            marker_corners, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
                image=self.frame_in,
                board=self.board,
                detectedCorners=marker_corners,
                detectedIds=marker_ids,
                rejectedCorners=rejected,
                # Known bug with refineDetectedMarkers, fixed in OpenCV 4.9: https://github.com/opencv/opencv/pull/24139
                cameraMatrix=self.camera_matrix if cv2.getVersionMajor() >= 4 and cv2.getVersionMinor() >= 9 else None,
                distCoeffs=self.dist_coeffs)

        if marker_ids is not None:
            self.markers_coords = np.array(marker_corners)[:, 0, :, :]
            self.marker_ids = marker_ids[:, 0]
            self.nb_markers = self.markers_coords.shape[0]

    def detect_corners(self, refine=True):

        self.points_coords = np.array([])
        self.points_ids = np.array([])
        self.nb_points = 0

        # If any marker has been detected, try to detect the board corners
        if self.nb_markers > 1:
            nb_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=self.markers_coords,
                markerIds=self.marker_ids,
                image=self.frame_in,
                board=self.board,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                minMarkers=1)
            if refine:
                try:
                    # Refine the board corners
                    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
                    charuco_corners = cv2.cornerSubPix(self.frame_in, charuco_corners,
                                                       winSize=(20, 20),
                                                       zeroZone=(-1, -1),
                                                       criteria=crit)
                except:
                    pass

            if charuco_corners is not None:
                self.points_coords = charuco_corners[:, 0, :]
                self.points_ids = charuco_ids[:, 0]
                self.nb_points = self.points_coords.shape[0]

    def reproject(self):

        self.reprojected_points = np.array([])
        self.nb_reprojected = 0

        self.reprojected_corners = np.array([])

        self.rvec = np.array([])
        self.tvec = np.array([])

        if self.camera_matrix is not None and self.nb_points > 6:
            _, rvec, tvec, error = cv2.solvePnPGeneric(self.objpoints[self.points_ids],
                                                       self.points_coords,
                                                       self.camera_matrix,
                                                       self.dist_coeffs)
            imgpoints, _ = cv2.projectPoints(self.objpoints, rvec[0], tvec[0], self.camera_matrix, self.dist_coeffs)

            self.curr_error_px = error[0][0]
            self.curr_error_mm = proj_geom.perspective_function(error[0][0], self.camera_matrix, tvec[0])

            self.reprojected_points = imgpoints[:, 0, :]
            self.nb_reprojected = self.reprojected_points.shape[0]

            board_corners_2d, _ = cv2.projectPoints(self.board_corners_3d, rvec[0], tvec[0], self.camera_matrix, self.dist_coeffs)
            self.reprojected_corners = board_corners_2d[:, 0, :]

            self.rvec = rvec[0]
            self.tvec = tvec[0]

    def update_coverage(self):

        if self.nb_points > 0:

            # Compute image area with detection
            detected_area = np.zeros(self.frame_in.shape[:2], dtype=np.uint8)
            pts = cv2.convexHull(np.round(self.points_coords[np.newaxis, ...]).astype(int))
            detected_area = cv2.fillPoly(detected_area, [pts], (255, 255, 255)).astype(bool)

            # Newly detected area is the union of the current detection and the inverse of the overlap with existing
            overlap = np.logical_and(detected_area, self.coverage)
            new_area = np.logical_and(detected_area, ~overlap)

            # If the new frame brings sufficient new coverage, add its data to the list
            if new_area.mean() * 100 >= 0.2 and self.nb_points > 5:
                self.multi_samples_points_coords.append(self.points_coords[np.newaxis, ...])
                self.multi_samples_points_ids.append(self.points_ids[np.newaxis, ...])

                self.coverage[detected_area] = True

    def calibrate(self):

        nb_samples = len(self.multi_samples_points_ids)

        # Compute calibration using all the frames we selected
        calib_ret, camera_matrix_new, dist_coeffs_new, rvecs_new, tvecs_new = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=self.multi_samples_points_coords,
            charucoIds=self.multi_samples_points_ids,
            board=self.board,
            imageSize=self.frame_in.shape[:2],
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            flags=cv2.CALIB_USE_QR)

        multi_objpoints = [self.objpoints[ids] for ids in self.multi_samples_points_ids]
        mean_error_px = 0.0
        for i in range(len(multi_objpoints)):
            _, rvec, tvec, error = cv2.solvePnPGeneric(multi_objpoints[i], self.multi_samples_points_coords[i], camera_matrix_new, dist_coeffs_new)
            mean_error_px += error[0][0]
        avg_err_px = mean_error_px / nb_samples

        if avg_err_px < self.best_error_px:
            self.camera_matrix = camera_matrix_new
            self.dist_coeffs = dist_coeffs_new
            self.best_error_px = avg_err_px

    def reset_samples(self):

        self.coverage.fill(False)
        self.coverage_overlay.fill(0)

        self.multi_samples_points_coords.clear()
        self.multi_samples_points_ids.clear()
        # self.multi_samples_frames_numbers.clear()

    def load_frame(self, frame):
        if frame.ndim == 3:
            self.frame_in = frame
        else:
            self.frame_in = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if self.coverage is None or self.coverage_overlay is None:
            self.coverage = np.full(self.frame_in.shape[:2], False, dtype=bool)
            self.coverage_overlay = np.zeros((*self.frame_in.shape[:2], 3), dtype=np.uint8)

    def visualise(self):

        frame_out = np.copy(self.frame_in)

        # If corners have been found, show them as red dots
        for xy in self.points_coords:
            frame_out = cv2.circle(frame_out, np.round(xy).astype(int), 2, (0, 0, 255), 2)

        # Display reprojected points: currently detected corners as yellow dots, the others as white dots
        for i, xy in enumerate(self.reprojected_points):
            if i in self.points_ids:
                frame_out = cv2.circle(frame_out, np.round(xy).astype(int), 2, (0, 255, 255), 2)
            else:
                frame_out = cv2.circle(frame_out, np.round(xy).astype(int), 2, (255, 255, 255), 2)

        # Display board corners in purple
        # for xy in self.reprojected_corners:
        #     frame_out = cv2.circle(frame_out, np.round(xy).astype(int), 4, (255, 0, 255), 4)

        # Display board perimeter in purple
        if len(self.reprojected_corners) > 0:
            pts = np.round(self.reprojected_corners).astype(int)
            frame_out = cv2.polylines(frame_out, [pts], True, (255, 0, 255), 2)

        # Add the coverage as a green overlay
        self.coverage_overlay[self.coverage, 1] = 255
        frame_out = cv2.addWeighted(frame_out, 0.85, self.coverage_overlay, 0.15, 0)

        # Undistort image
        if self.camera_matrix is not None:
            h, w = self.frame_in.shape[:2]
            optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h))
            frame_out = cv2.undistort(frame_out, self.camera_matrix, self.dist_coeffs, None, optimal_camera_matrix)

        # Add information text to the visualisation image
        frame_out = cv2.putText(frame_out,
                                     f"Aruco markers: {self.nb_markers}/{self.total_markers}", (30, 30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        frame_out = cv2.putText(frame_out,
                                     f"Corners: {self.nb_points}/{self.total_points}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        frame_out = cv2.putText(frame_out,
                                     f"Area: {self.coverage.mean() * 100:.2f}% ({len(self.multi_samples_points_coords)} snapshots)", (30, 90),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        txt = f"{self.curr_error_px:.2f} px ({self.curr_error_mm:.3f} mm)" if (self.best_error_px != float('inf')) & (self.curr_error_mm != float('inf')) else '-'
        frame_out = cv2.putText(frame_out,
                                     f"Current reprojection error: {txt}",
                                     (30, 120),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        txt = f"{self.best_error_px:.2f} px" if self.best_error_px != float('inf') else '-'
        frame_out = cv2.putText(frame_out,
                                     f"Best average reprojection error: {txt}",
                                     (30, 150),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return frame_out

    def detect(self):

        if self.frame_in is not None:
            self.detect_markers(refine=True)
            self.detect_corners(refine=True)

            self.update_coverage()

            self.reproject()

            if self.coverage.mean() >= 0.6 and len(self.multi_samples_points_ids) >= self.min_samples:
                self.calibrate()
                self.reset_samples()

    def save(self, filepath):

        if self.camera_matrix is not None and self.camera_matrix is not None:

            filepath = Path(filepath)
            if not filepath.suffix == '.toml':
                filepath = filepath.parent / f'{filepath.stem}.toml'

            d = {'camera_matrix': self.camera_matrix.tolist(), 'dist_coeffs': self.dist_coeffs.tolist()}

            with open(filepath, 'w') as f:
                # Remove trailing commas
                toml_str = toml.dumps(d).replace(',]', ' ]')
                # Add indents (yes this one-liner is atrocious)
                lines = [l.replace('], [', f'],\n{"".ljust(len(l.split("=")[0]) + 4)}[') for l in toml_str.splitlines()]
                toml_str_formatted = '\n'.join(lines)
                f.write(toml_str_formatted)

    def load(self, filepath):

        filepath = Path(filepath)
        if not filepath.suffix == '.toml':
            filepath = filepath.parent / f'{filepath.stem}.toml'

        d = toml.load(filepath)

        self.camera_matrix = np.array(d['camera_matrix'])
        self.dist_coeffs = np.array(d['dist_coeffs'])


## -------------------------------------------------------------

if __name__ == '__main__':
    # Run mini-GUI

    calib = IntrinsicsTool(charuco_board)

    w_name = 'detection'
    cv2.namedWindow(w_name)

    # Load video
    video_path = FOLDER / FILE
    _, nb_frames = utilities.probe_video(video_path)

    cap = cv2.VideoCapture(video_path.as_posix())


    def on_slider_change(trackbar_value):
        cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
        r, frame = cap.read()
        if r:
            calib.load_frame(frame)
            calib.detect()
            frame_out = calib.visualise()
            cv2.imshow(w_name, frame_out)
        pass

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

    # Save calibration data to disk if the reprojection error is ok
    if calib.best_error_px < REPROJ_ERR:
        calib.save(FOLDER / Path(FILE).stem)
        print(f"Calibration saved!")
