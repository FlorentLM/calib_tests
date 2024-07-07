import numpy as np
import cv2
from PIL import Image
from collections import deque
from pathlib import Path
np.set_printoptions(precision=3, suppress=True, threshold=5)


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
#

# Board parameters to detect

BOARD_COLS = 7                      # Total rows in the board (chessboard)
BOARD_ROWS = 10                     # Total cols in the board
SQUARE_LENGTH_MM = 5                # Length of one chessboard square in real life units (i.e. mm)
MARKER_BITS = 4                     # Size of the markers in 'pixels' (not really, but you get the idea)

FOLDER = Path(f'/Users/florent/Desktop/cajal_messor_videos/calibration/')
CAM = 'cam3'

SAVE = True

##

def lines_intersection(line1, line2):
    """
        Returns the intersection point of two lines in 2D
    """
    xdiff, ydiff = -np.vstack((np.diff(line1, axis=0), np.diff(line2, axis=0))).T

    div = np.cross(xdiff, ydiff)
    if div == 0:
       raise Exception('These lines do not intersect!')

    d = (np.cross(*line1), np.cross(*line2))

    return np.cross(d, xdiff) / div, np.cross(d, ydiff) / div


def reduce_polygon(arr, nb_sides=4):
    """
        Simplifies a polygon to the given number of sides, by iteratively removing the shortest side
    """
    nb_edges = arr.shape[0]

    # Compute all lengths and coords for starting polygon
    rolled = np.roll(arr, -1, axis=0)
    diffs = rolled - arr
    sides_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    sides_coords = np.hstack((arr, rolled)).reshape(-1, 2, 2).astype(np.float32)

    while nb_edges > nb_sides:
        # Remove the shortest side
        idx_shortest = np.argmin(sides_lengths)

        # Replace the vertex coordinates in the two adjacent sides
        idx_adj_1 = idx_shortest - 1
        idx_adj_2 = idx_shortest + 1

        # Loop over if last edge is the shortest
        if idx_adj_2 == nb_edges:
            idx_adj_2 = 0

        new_vertex = lines_intersection(sides_coords[idx_adj_1], sides_coords[idx_adj_2])
        sides_coords[idx_adj_1, 1, :] = new_vertex
        sides_coords[idx_adj_2, 0, :] = new_vertex

        # update the two new lengths
        sides_lengths[idx_adj_1] = np.sqrt(np.sum(np.diff(sides_coords[idx_adj_1, :, :], axis=0) ** 2))
        sides_lengths[idx_adj_2] = np.sqrt(np.sum(np.diff(sides_coords[idx_adj_2, :, :], axis=0) ** 2))

        # Update the coord and legths arrays
        sides_coords = np.delete(sides_coords, idx_shortest, axis=0)
        sides_lengths = np.delete(sides_lengths, idx_shortest, axis=0)
        nb_edges = sides_coords.shape[0]

    return np.round(sides_coords[:, 0, :]).astype(int)


def generate_charuco(board_rows, board_cols, square_length_mm=5, marker_bits=4, save_svg=True):
    """
        Generates a Charuco board for the given parameters, and optionally saves it in a SVG file.
    """
    all_dict_sizes = [50, 100, 250, 1000]

    mk_l_px = marker_bits + 2
    sq_l_px = mk_l_px + 2

    marker_length_mm = mk_l_px / sq_l_px * square_length_mm

    dict_size = next(s for s in all_dict_sizes if s >= board_rows * board_cols)
    dict_name = f'DICT_{marker_bits}X{marker_bits}_{dict_size}'

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((board_cols, board_rows),  # number of chessboard squares in x and y directions
                                   square_length_mm,  # chessboard square side length (normally in meters)
                                   marker_length_mm,  # marker side length (same unit than squareLength)
                                   aruco_dict)

    if save_svg:
        chessboard_arr = (~np.indices((board_rows, board_cols)).sum(axis=0) % 2).astype(bool)

        svg_lines = [f'<svg version="1.1" width="100%" height="100%" viewBox="0 0 {sq_l_px * board_cols} {sq_l_px * board_rows}" xmlns="http://www.w3.org/2000/svg">']
        svg_lines.append('  <g id="charuco">')

        # Chessboard group
        svg_lines.append('    <g id="chessboard">')
        cc, rr = np.where(chessboard_arr)
        for i, rc in enumerate(zip(rr, cc)):
            svg_lines.append(f'      <rect id="{i}" x="{rc[0] * sq_l_px}" y="{rc[1] * sq_l_px}" width="{sq_l_px}" height="{sq_l_px}" fill="#000000"/>')
        svg_lines.append('    </g>')

        # Aruco markers group
        svg_lines.append('    <g id="aruco_markers">')
        cc, rr = np.where(~chessboard_arr)
        for i, rc in enumerate(zip(rr, cc)):
            marker = aruco_dict.generateImageMarker(i, mk_l_px, mk_l_px).astype(bool)
            py, px = np.where(marker)
            svg_lines.append(f'      <g id="{i}">')
            svg_lines.append(
                f'        <rect x="{rc[0] * sq_l_px + 1}" y="{rc[1] * sq_l_px + 1}" width="{mk_l_px}" height="{mk_l_px}" fill="#000000"/>')
            for x, y in zip(px, py):
                svg_lines.append(f'        <rect x="{rc[0] * sq_l_px + x + 1}" y="{rc[1] * sq_l_px + y + 1}" width="1" height="1" fill="#ffffff"/>')

            svg_lines.append('      </g>')

        svg_lines.append('    </g>')
        svg_lines.append('  </g>')
        svg_lines.append('</svg>')

        filename = f'Charuco{board_rows}x{board_cols}_markers{marker_bits}x{marker_bits}_{dict_size}'

        with open(f'{filename}.svg', 'w') as f:
            f.write('\n'.join(svg_lines))
            # TODO - save a A4 page with many copies of the board in different sizes

    return aruco_dict, board

##

# Generate Charuco board and corresponding detector
aruco_dict, charuco_board = generate_charuco(BOARD_ROWS, BOARD_COLS,
                                             square_length_mm=SQUARE_LENGTH_MM,
                                             marker_bits=MARKER_BITS,
                                             save_svg=False)

detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Stop criteria for board corners refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# Initialise these to None for first estimation without a prior
camera_matrix = None
dist_coeffs = None

##

video_path = FOLDER / f'{CAM}.mp4'

# Read one frame to get dimensions
cap = cv2.VideoCapture(video_path.as_posix())
r, frame = cap.read()
nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Init some global variables

img_viz = np.copy(frame)

empty_frame = np.zeros_like(frame)
source_shape = np.array(frame.shape, dtype=np.uint32)

covered_area_so_far = np.zeros(source_shape[:2], dtype=np.uint8)
current_coverage_pct = 0
all_frames_corners = deque()
all_frames_ids = deque()
all_frames_frame_ids = deque()

nb_markers = len(charuco_board.getIds())
nb_corners = len(charuco_board.getChessboardCorners())

previous_poly = None

w_name = 'detection'

# This is the function that does the bulk of the work
def detect(frame, frame_id=None):
    img_viz = np.copy(frame)
    empty_frame = np.zeros_like(frame)
    nb_detected_squares = 0

    # Detect and refine aruco markers
    marker_corners, marker_ids, rejected = detector.detectMarkers(frame)
    marker_corners, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
        image=frame,
        board=charuco_board,
        detectedCorners=marker_corners,
        detectedIds=marker_ids,
        rejectedCorners=rejected)

    if marker_ids is None:
        nb_detected_markers = 0
        this_frames_area = 0

    # If any marker has been detected, detect the board corners
    else:
        nb_detected_markers = len(marker_ids)

        markers_col = cv2.aruco.drawDetectedMarkers(np.copy(empty_frame), marker_corners)
        markers = markers_col[:, :, 1]
        corners = markers_col[:, :, 2]
        markers_full_perimeter = np.array(~((markers.astype(bool) & corners.astype(bool)) | markers.astype(bool))).astype(np.uint8) * 255

        mask = np.zeros(source_shape[:2] + 2, dtype=np.uint8)
        _, markers_filled, _, _ = cv2.floodFill(markers_full_perimeter, mask, (0, 0), 0)

        contours, _ = cv2.findContours(markers_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.vstack(contours)

        hull = np.array(cv2.convexHull(contours))[:, 0, :]
        new_hull = reduce_polygon(hull, nb_sides=8)    # 8 is a good value to get rectangle-ish polygons most of the time and still avoid jumpy triangles when the board is viewed from the side

        board_area = np.array(~cv2.drawContours(np.copy(empty_frame),[new_hull.astype(int)], 0, (255, 255, 255), 1)[:, :, 0].astype(bool), dtype=np.uint8) * 255
        mask = np.zeros(source_shape[:2] + 2, dtype=np.uint8)
        _, this_frames_area, _, _ = cv2.floodFill(board_area, mask, (0, 0), 0)

        nb_detected_squares, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=frame,
            board=charuco_board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            minMarkers=0)
        try:
            charuco_corners = cv2.cornerSubPix(frame, charuco_corners,
                                               winSize=(20, 20),
                                               zeroZone=(-1, -1),
                                               criteria=criteria)
        except:
            pass

        # If corners have been found, compute the area covered by this frame's board detection
        if charuco_corners is not None:
            # the area that overlaps with the area covered so far
            curr_overlap_area = cv2.bitwise_and(this_frames_area, covered_area_so_far)
            newly_added_area = cv2.bitwise_and(this_frames_area, cv2.bitwise_not(curr_overlap_area))

            new_area_pct = newly_added_area.astype(bool).sum() / np.prod(this_frames_area.shape) * 100

            # If the newly added area is more than 1% of the image, store the current detection
            if new_area_pct >= 0.2 and len(charuco_corners) > 5:
                all_frames_corners.append(charuco_corners)
                all_frames_ids.append(charuco_ids)
                covered_area_so_far[this_frames_area.astype(bool)] = 255

                if frame_id is not None:
                    all_frames_frame_ids.append(frame_id)

            # curr_added_area_viz = cv2.cvtColor(newly_added_area, cv2.COLOR_GRAY2BGR)
            # alpha = 0.25
            # img_viz = cv2.addWeighted(img_viz, 1 - alpha, curr_added_area_viz, alpha, 0)

            # Show corners as red dots
            for xy in charuco_corners[:, 0]:
                img_viz = cv2.circle(img_viz, np.round(xy).astype(int), 2, (0, 0, 255), 2)

    curr_added_area_viz = cv2.cvtColor(covered_area_so_far, cv2.COLOR_GRAY2BGR)
    curr_added_area_viz[:, :, 2] = 0
    curr_added_area_viz[:, :, 0] = 0
    alpha = 0.15
    img_viz = cv2.addWeighted(img_viz, 1 - alpha, curr_added_area_viz, alpha, 0)

    current_coverage_pct = covered_area_so_far.astype(bool).sum() / np.prod(img_viz.shape) * 100

    # Add texts
    img_viz = cv2.putText(img_viz, f"Aruco markers: {nb_detected_markers}/{nb_markers}", (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), thickness=2)

    img_viz = cv2.putText(img_viz, f"Corners: {nb_detected_squares}/{nb_corners}", (30, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), thickness=2)

    img_viz = cv2.putText(img_viz, f"Area: {current_coverage_pct:.2f}% ({len(all_frames_corners)} snapshots)", (30, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), thickness=2)

    cv2.imshow(w_name, img_viz)

##

# Run mini GUI
def on_slider_change(trackbar_value):
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
    r, frame = cap.read()
    if r:
        detect(frame, frame_id=trackbar_value)
    pass

cv2.namedWindow(w_name)
cv2.createTrackbar('Frame:', w_name, 0, nb_frames, on_slider_change)
on_slider_change(0)

while True:
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()

##

# Compute calibration using all the frames we selected
retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_frames_corners,
    charucoIds=all_frames_ids,
    board=charuco_board,
    imageSize=source_shape[:2],
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
    print(f"Frame {all_frames_frame_ids[i]} error: {error}")
avg_err = mean_error / len(objpoints)
print(f"Average error: {avg_err}")

# Save calibration data to disk if the reprojection error is ok

if SAVE:
    if avg_err < 0.2:
        print(f"Calibration successful! Saving.")
        np.savez_compressed(FOLDER / f'{CAM}_frames_ids.npz', np.array(list(all_frames_frame_ids)))
        np.savez_compressed(FOLDER / f'{CAM}_camera_matrix.npz', camera_matrix)
        np.savez_compressed(FOLDER / f'{CAM}_dist_coeffs.npz', dist_coeffs)
        np.savez_compressed(FOLDER / f'{CAM}_rvecs.npz', np.hstack(rvecs).T)
        np.savez_compressed(FOLDER / f'{CAM}_tvecs.npz', np.hstack(tvecs).T)
    else:
        print(f"Calibration is meh... Not saving")
