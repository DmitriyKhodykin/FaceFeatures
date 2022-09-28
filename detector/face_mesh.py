import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


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
