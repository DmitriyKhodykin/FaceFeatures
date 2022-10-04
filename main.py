import cv2
import mediapipe as mp
import cvzone
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

                frame_h, frame_w = frame.shape[:2]

                # Overlaying the image of glasses.
                glasses_img = cv2.imread("static/glasses.png", cv2.IMREAD_UNCHANGED)
                
                if results.multi_face_landmarks:

                    all_faces_keypoints = CustomDrawingUtils(frame).get_glasses_coordinates("upper_edge")
                    glasses_img = ImageTransforming(glasses_img).scaling_image(50)
                    result_frame = cvzone.overlayPNG(frame, glasses_img, [220, 180])

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow("Glasses", result_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Main()
    app.exec_pipeline()