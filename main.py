import cv2
import mediapipe as mp

from detector.face_mesh import DrawingUtils
from utils.imtransformer import ImageTransforming


def _keypoints(landmark_list):
    keypoints = []

    for data_point in landmark_list.landmark:
        keypoints.append({
            "x": data_point.x,
            "y": data_point.y,
            "z": data_point.z
        })
    return keypoints


class Main:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

    def exec_pipeline(self):
        drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        cap = cv2.VideoCapture(0)
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
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
                    for face_landmarks in results.multi_face_landmarks:
                        kp = _keypoints(face_landmarks)
                        print(kp[0]["x"])
                        du = DrawingUtils(image)
                        # du.drawing_landmarks(face_landmarks, "tesselation")
                        # du.drawing_landmarks(face_landmarks, "contours")
                        du.drawing_landmarks(face_landmarks, "irises")

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow("MediaPipe Face Mesh", cv2.flip(image, 1))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()


if __name__ == "__main__":
    app = Main()
    app.exec_pipeline()
