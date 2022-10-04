import cv2
import mediapipe as mp
import cvzone
import numpy as np

from detector.face_mesh import FaceModel
from recognizer.face_recognizer import CustomDrawingUtils
from utils.imtransformer import ImageTransforming


class Main:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

    def exec_pipeline(self):

        capture = cv2.VideoCapture(0)

        with FaceModel().create_facemesh_model() as face_mesh:
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance changing img color.
                imt = ImageTransforming(frame)
                imt.change_color(cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)
                imt.change_color(cv2.COLOR_RGB2BGR)

                # Overlaying the image of glasses.
                result_frame = None
                glasses_img = cv2.imread(
                    "static/glasses.png", cv2.IMREAD_UNCHANGED
                )  # Unchanged - with alpha layer

                if results.multi_face_landmarks:

                    all_faces_keypoints = CustomDrawingUtils(
                        frame
                    ).get_glasses_coordinates("upper_edge")

                    glasses_img = ImageTransforming(glasses_img).scaling_image(50)

                    try:
                        result_frame = cvzone.overlayPNG(
                            frame,
                            glasses_img,
                            [
                                all_faces_keypoints[0][0][0] - 50,
                                all_faces_keypoints[0][1][1] - 20,
                            ],
                        )
                    except Exception as error:
                        print(error)
                        continue

                # Flip the image horizontally for a selfie-view display.
                fliped_frame = cv2.flip(result_frame, 1)
                
                cv2.imshow("Glasses", fliped_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Main()
    app.exec_pipeline()
