import cv2
import mediapipe as mp
import numpy as np

from detector.face_mesh import FaceModel
from detector.face_mesh import CustomDrawingUtils
from utils.imtransformer import ImageTransforming


class Main:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

    def exec_pipeline(self):

        capture = cv2.VideoCapture(0)

        with FaceModel().create_facemesh_model() as face_mesh:
            while capture.isOpened():
                success, image = capture.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance changing img color.
                imt = ImageTransforming(image)
                imt.change_color(cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                imt.change_color(cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:

                    glases_coordinates = CustomDrawingUtils(
                        image
                    ).get_glases_coordinates()

                    cv2.polylines(
                        image,
                        glases_coordinates,
                        True,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow("ITentika glases", cv2.flip(image, 1))

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Main()
    app.exec_pipeline()
