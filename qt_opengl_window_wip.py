import sys

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QWidget, QPushButton, QDialog
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QSurfaceFormat, QOpenGLContext, QPainter, QPolygon, QColor

import OpenGL.GL as gl
import OpenGL.GLU as glu

from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True, threshold=5)
import cv2

import proj_geom
import utilities
from intrinsics import CalibrationTool


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, charuco_board, full_vids, FOLDER, parent=None):
        super().__init__(parent=parent)

        self._angle_x = 0
        self._angle_y = 0
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._zoom = -500.0

        self._mouse_last_pos = QPoint()

        self.frame_index = 0
        self.charuco_board = charuco_board
        self.full_vids = full_vids
        self.FOLDER = FOLDER

        self.calib = CalibrationTool(self.charuco_board)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_view)
        self.timer.start(30)

        self.elements = []
        self.other_elems = []

        # cube = {
        #     'mode': 'lines',
        #     'vertices': [
        #         # Top
        #         [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],    # RB, LB
        #         [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0],      # LF, RF
        #
        #         # Bottom
        #         [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0],    # RF, LF
        #         [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]   # LB, RB
        #     ],
        #     'edges': [
        #         [0, 1], [1, 2], [2, 3], [3, 0],         # Top
        #         [4, 5], [5, 6], [6, 7], [7, 4],         # Bottom
        #         [0, 7], [1, 6], [2, 5], [3, 4],         # Side connections
        #     ]
        # }
        #
        # points = {
        #     'mode': 'points',
        #     'vertices': cube['vertices'],
        #     'size': 10.0,
        #     'colour': (0.3, 0.8, 0.8),
        # }
        #
        # sphere = {
        #     'mode': 'sphere',
        #     'centre': [0.0, 0.0, 0.0],
        #     'radius': 0.1,
        #     'colour': (0.0, 1.0, 0.0)
        # }

        # self.elements.append(cube)
        # self.elements.append(points)
        # self.elements.append(sphere)

    def initializeGL(self):
        self.context = QOpenGLContext(self)

        # Create an OpenGL 2.1 context - becaue we don't need any shaders so fixed pipeline is best
        format = QSurfaceFormat()
        format.setVersion(2, 1)
        format.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
        self.context.setFormat(format)
        if not self.context.create():
            raise Exception("Unable to create GL context")

        # gl.glClearColor(0.5, 0.8, 0.7, 1.0)        # Background colour
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)        # Background colour
        # gl.glEnable(gl.GL_DEPTH_TEST)                                     # Depth test is needed
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)        # Blend function for transparency
        gl.glEnable(gl.GL_BLEND)                                          # Blend transparency
        gl.glEnable(gl.GL_POINT_SMOOTH)                                   # Round points

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45.0, w / h, 0.1, 1000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # move and rotate the GL camera according to user input
        gl.glTranslatef(self._pos_x, self._pos_y, self._zoom)
        gl.glRotatef(self._angle_x, 1.0, 0.0, 0.0)
        gl.glRotatef(self._angle_y, 0.0, 1.0, 0.0)

        # and finally draw all the stuff
        self.render_elements()

    def render_elements(self):

        for obj in self.elements + self.other_elems:

            if obj.get('opaque', False):
                gl.glEnable(gl.GL_DEPTH_TEST)
            else:
                gl.glDisable(gl.GL_DEPTH_TEST)

            match obj['mode']:
                case 'lines':
                    gl.glLineWidth(obj.get('linewidth', 1.0))
                    gl.glBegin(gl.GL_LINES)
                    gl.glColor4f(*obj.get('colour', (1.0, 1.0, 1.0, 1.0)))
                    for e in obj['edges']:
                        for v in e:
                            gl.glVertex3fv(obj['vertices'][v])
                    gl.glEnd()

                case 'quads':
                    gl.glBegin(gl.GL_QUADS)
                    gl.glColor4f(*obj.get('colour', (1.0, 1.0, 1.0, 1.0)))
                    for e in obj['edges']:
                        for v in e:
                            gl.glVertex3fv(obj['vertices'][v])
                    gl.glEnd()

                case 'points':
                    gl.glPointSize(obj.get('size', 1.0))
                    gl.glColor4f(*obj.get('colour', (1.0, 1.0, 1.0, 1.0)))
                    gl.glBegin(gl.GL_POINTS)
                    for v in obj['vertices']:
                        gl.glVertex3f(*v)
                    gl.glEnd()

                case 'sphere':
                    gl.glColor4f(*obj.get('colour', (1.0, 1.0, 1.0, 1.0)))
                    gl.glPushMatrix()
                    gl.glTranslatef(*obj['centre'])
                    gluQuadric = glu.gluNewQuadric()
                    glu.gluSphere(gluQuadric, obj.get('radius', 1.0), 64, 64)
                    glu.gluDeleteQuadric(gluQuadric)
                    gl.glPopMatrix()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._mouse_last_pos
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._pos_x += delta.x()
                self._pos_y -= delta.y()
            else:
                self._angle_x += delta.y() * 0.5
                self._angle_y += delta.x() * 0.5
            self._mouse_last_pos = event.position()
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self._zoom += delta * 0.1
        self.update()

    def _update_view(self):
        self.update()

    def change_frame(self, frame_index):
        self.frame_index = frame_index
        self.load_frame()
        self.generate_models()

    def load_frame(self):

        rvecs = {}
        tvecs = {}
        camera_matrices = {}
        distortion_coeffs = {}

        recentre = False
        origin = None

        multi_cams_coords = {}
        multi_frustum_coords = {}

        for name, vid in self.full_vids.items():

            self.calib.load(self.FOLDER / f'{name}.toml')

            vid.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            r, frame = vid.read()
            self.calib.load(self.FOLDER / f'{name}.toml')

            self.calib.detect(frame)
            self.calib.reproject()

            if not self.calib.has_extrinsics:
                return None, None, None

            rvecs[name] = self.calib.rvec
            tvecs[name] = self.calib.tvec
            camera_matrices[name] = self.calib.camera_matrix
            distortion_coeffs[name] = self.calib.dist_coeffs

            h, w = frame.shape[:2]
            points2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

            if recentre:
                camera_matrices[name][: 2, 2] = frame.shape[1] / 2.0, frame.shape[0] / 2.0

            # Extrinsics matrix in object-space
            ext_mat = proj_geom.extrinsics_mat(rvecs[name], tvecs[name], hom=True)
            inv_mat = np.linalg.inv(ext_mat)

            if origin is None:
                origin = ext_mat

            transform_origin_mat = np.dot(origin, inv_mat)

            # The T part of the inverse of the transformation matrix is the camera's position
            cam_point3d = transform_origin_mat[:3, 3]

            # Plot frustum to far point = depth
            frustum_points3d = proj_geom.back_projection(points2d, 130, camera_matrices[name], transform_origin_mat, invert=False)

            multi_cams_coords[name] = cam_point3d
            multi_frustum_coords[name] = frustum_points3d

        # Charuco board
        board_points3d = self.calib.corners3d - inv_mat[:3, 3]
        board_points3d = np.dot(board_points3d, inv_mat[:3, :3])

        return multi_cams_coords, multi_frustum_coords, board_points3d

    def generate_models(self):

        colors = [(0.9, 0.3, 0.3, 1.0), (0.3, 0.9, 0.3, 1.0), (0.3, 0.3, 0.9, 1.0)]

        multi_cams_coords, multi_frustum_coords, board_points3d = self.load_frame()

        if board_points3d is None:
            self.other_elems = []

        else:

            self.elements = []
            self.other_elems = []

            for n, name in enumerate(self.full_vids.keys()):
                frustum_lines = {
                    'mode': 'lines',
                    'opaque': True,
                    'vertices': [p.tolist() for p in np.vstack([multi_cams_coords[name], multi_frustum_coords[name]])],
                    'edges': [
                        # [1, 2], [2, 3], [3, 4], [4, 1],  # Plan
                        [0, 1], [0, 2], [0, 3], [0, 4],  # Sides
                    ],
                    'colour': colors[n],
                    'linewidth': 1.0
                }
                self.elements.append(frustum_lines)

                frustum_plan = {
                    'mode': 'quads',
                    'vertices': [p.tolist() for p in multi_frustum_coords[name]],
                    'edges': [
                        [0, 1], [1, 2], [2, 3], [3, 0],  # Plan
                    ],
                    'colour': (*colors[n][:3], 0.05),
                }
                self.elements.append(frustum_plan)

                frustum_plan_perimeter = {
                    'mode': 'lines',
                    'opaque': True,
                    'vertices': [p.tolist() for p in multi_frustum_coords[name]],
                    'edges': [
                        [0, 1], [1, 2], [2, 3], [3, 0],  # Plan
                    ],
                    'colour': colors[n],
                    'linewidth': 2.0
                }
                self.elements.append(frustum_plan_perimeter)

                cam = {
                    'mode': 'points',
                    'opaque': True,
                    'vertices': [multi_cams_coords[name].astype(np.float32).tolist()],
                    'size': 30.0,
                    'colour': colors[n],
                }
                self.elements.append(cam)

            board = {
                'mode': 'quads',
                'vertices': [p.tolist() for p in board_points3d.astype(np.float32)],
                'edges': [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Plan
                ],
                'colour': (0.1, 0.1, 0.1, 0.1),
            }
            self.other_elems.append(board)

            board_perimeter = {
                'mode': 'lines',
                'vertices': [p.tolist() for p in board_points3d.astype(np.float32)],
                'edges': [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Plan
                ],
                'colour': (1.0, 1.0, 1.0, 0.1),
            }
            self.other_elems.append(board_perimeter)

class MinimalWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Minimal OpenGL window")
        self.setGeometry(100, 100, 800, 600)

        BOARD_COLS = 7  # Total rows in the board (chessboard)
        BOARD_ROWS = 10  # Total cols in the board
        SQUARE_LENGTH_MM = 5  # Length of one chessboard square in real life units (i.e. mm)
        MARKER_BITS = 4  # Size of the markers in 'pixels' (not really, but you get the idea)

        charuco_board = utilities.generate_charuco(board_rows=BOARD_ROWS,
                                                   board_cols=BOARD_COLS,
                                                   square_length_mm=SQUARE_LENGTH_MM,
                                                   marker_bits=MARKER_BITS)

        FOLDER = Path('/Users/florent/Desktop/cajal_messor_videos/calibration')

        full_vids = {}

        vid_lengths = []
        for f in FOLDER.glob('*.mp4'):
            shape, nb_frames = utilities.probe_video(f)
            cap = cv2.VideoCapture(f.as_posix())
            vid_lengths.append(nb_frames)

            # print(f"[INFO] Loading {f.name} to memory", flush=False)
            # all_frames = []
            # while cap.isOpened():
            #     r, frame = cap.read()
            #     if r:
            #         all_frames.append(frame)
            #     else:
            #         break
            # cap.release()
            # print(f"Done")

            # full_vids[f.stem] = all_frames
            full_vids[f.stem] = cap

        nb_frames = min(vid_lengths)

        central_widget = QWidget()
        central_widget_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(nb_frames)
        self.slider.setSingleStep(1)
        self.slider.setValue(4070)
        self.slider.setTracking(True)

        central_widget_layout.addWidget(self.slider)

        self.opengl_widget = OpenGLWidget(charuco_board, full_vids, FOLDER)
        self.opengl_widget.change_frame(self.slider.value())

        self.slider.valueChanged.connect(self.opengl_widget.change_frame)

        central_widget_layout.addWidget(self.opengl_widget)

        self.button = QPushButton("Pop-up")
        self.button.setGeometry(100, 80, 100, 30)
        self.button.clicked.connect(self.show_popup)

        central_widget_layout.addWidget(self.button)

    def show_popup(self):
        self.popup = PopUpDialog(parent=self)
        button_rect = self.button.geometry()
        button_pos = self.button.mapToGlobal(QPoint(0, 0))
        self.popup.move(button_pos.x() + button_rect.width() // 2 - self.popup.width() // 2,
                        button_pos.y() + button_rect.height())
        self.popup.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.slider.setValue(self.slider.value() - 1)
        elif event.key() == Qt.Key.Key_Right:
            self.slider.setValue(self.slider.value() + 1)

class PopUpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pop-up")
        self.setFixedSize(200, 100)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Popup)

        layout = QVBoxLayout()
        button1 = QPushButton("Button 1")
        button2 = QPushButton("Button 2")
        layout.addWidget(button1)
        layout.addWidget(button2)
        self.setLayout(layout)

    def leaveEvent(self, event):
        self.close()
        event.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MinimalWindow()
    window.show()

    sys.exit(app.exec())
