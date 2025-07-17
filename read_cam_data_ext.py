import mediapipe as mp
import cv2
import time

def read_camera_data(camera_index=4):
    cap =cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {camera_index}")
        return
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frames")
            continue


        imgRGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=hands.process(imgRGB)

        #print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    #print(id,lm)
                    h, w, c = frame.shape
                    cx, cy = int(lm.x*w),int(lm.y*h)
                    print(id, cx, cy)

                    #if id == 4:
                    #cv2.circle(frame,(cx,cy), 15, (255,0,255), cv2.FILLED)

                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1/(cTime - pTime)

        pTime = cTime

        cv2.putText(frame, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
        (255,0,255),3)


        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_camera_data()

