import mediapipe as mp
import cv2
import time

def read_camera_data():
    cap =cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frames")
            continue
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_camera_data()

