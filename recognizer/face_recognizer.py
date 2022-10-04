"""Module for drawing face mesh landmarks.

Landmarks (keypoints) defenition:
https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/
mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

Docs:
    * How to Overlay Transparent PNG Image Over Video: https://www.youtube.com/watch?v=1LfKNmOJgjw
    * PNG Overlay and Image Rotation: https://www.youtube.com/watch?v=voRFbl-GKGY
"""

import numpy as np

from detector.face_mesh import FaceModel


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
            try:
                glases_keypoints = [all_faces_keypoints[[71, 301, 346, 6, 117]]]
            except TypeError as error:
                print("FaceRecognizerError:", error)

        elif area == "upper_edge":
            try:
                glases_keypoints = [all_faces_keypoints[[71, 301]]]
            except TypeError as error:
                print("FaceRecognizerError:", error)

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
