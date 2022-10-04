""" Mediapipe module for drawing face mesh landmarks.

Landmarks (keypoints) defenition:
https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/
mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

Docs:
    * How to Overlay Transparent PNG Image Over Video: https://www.youtube.com/watch?v=1LfKNmOJgjw
    * PNG Overlay and Image Rotation: https://www.youtube.com/watch?v=voRFbl-GKGY
"""

import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


class CustomDrawingUtils:
    def __init__(self, image):
        self.image = image
        self.model = FaceModel().create_facemesh_model()
        self.results = self.model.process(image)

    def get_glasses_coordinates(self, area: str):
        """Returns the coordinates of the eye area in the face picture.
        area:
            * full - 5 coordinates from glasses area
            * upper_edge - 2 coordinates of upper edge of the eyebrows
        """
        all_faces_keypoints = self._get_face_keypoints()
        if area == "full":
            glases_keypoints = [all_faces_keypoints[[71, 301, 346, 6, 117]]]
        elif area == "upper_edge":
            glases_keypoints = [all_faces_keypoints[[71, 301]]]
        return glases_keypoints

    def _get_face_keypoints(self):
        img_h, img_w = self.image.shape[:2]

        if self.results.multi_face_landmarks:

            face_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in self.results.multi_face_landmarks[0].landmark
                ]
            )
            return face_points


class DrawingUtils:
    def __init__(self, image):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.image = image

    def drawing_landmarks(self, landmarks, mesh_type):
        """Draws different configurations of lines and dots on the face.
        mesh_type:
            FACEMESH_TESSELATION
            FACEMESH_CONTOURS
            FACEMESH_IRISES
        See: https://google.github.io/mediapipe/solutions/face_mesh.html
        """
        if mesh_type == "tesselation":
            self.mp_drawing.draw_landmarks(
                image=self.image,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
        elif mesh_type == "contours":
            self.mp_drawing.draw_landmarks(
                image=self.image,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
        elif mesh_type == "irises":
            self.mp_drawing.draw_landmarks(
                image=self.image,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return self


class FaceModel:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

    def create_facemesh_model(self):
        """Set model parameters.
        See: https://google.github.io/mediapipe/solutions/face_mesh.html
        """
        model = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return model
