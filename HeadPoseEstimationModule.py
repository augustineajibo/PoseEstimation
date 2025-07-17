import cv2
import mediapipe as mp
import numpy as np
import time

class headPose():
    def __init__(self, mode=False, maxFace=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxFace = maxFace
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFace_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mpFace_mesh.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawing_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFace(self, frame, draw=True):
        imgRGB = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
        #imgRGB.flags.writeable=False

        self.results = self.face_mesh.process(imgRGB)

        #imgRGB.flags.writeable=True

        #imgRGB = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = frame.shape
        face_3d = []
        face_2d = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        #Get the 2D Coordinates
                        face_2d.append([x, y])

                        #Get the 3D Coordinate
                        face_3d.append([x, y, lm.z])

                # Convert it to NumPy arrays
                face_2d = np.array(face_2d, dtype=np.float64).reshape(-1, 2)
                face_3d = np.array(face_3d, dtype=np.float64).reshape(-1, 3)

                # The camera matrix
                focal_length = 1 * img_w
                cam_metrix = np.array([
                    [focal_length, 0, img_h / 2],
                    [0, focal_length, img_w / 2],
                    [0, 0, 1]
                ], dtype=np.float64)

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_metrix, dist_matrix)

                #Get rotation matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                #Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                #Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                #See where the user's head is titling
                if y<=10:
                    text="Looking Left"
                elif y> 10:
                    text="Looking Right"
                elif x<-10:
                    text="Looking Down"
                elif x >10:
                    text="Looking Up"
                else:
                    text="Forward"

                #Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_metrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(frame,p1,p2,(255,0,0),3)


                #Add text to the image
                cv2.putText(frame,text,(20,50), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
                cv2.putText(frame,"x:" + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                cv2.putText(frame, "y:" + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(frame, "z:" + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                self.mpDraw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mpFace_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec

                )


def main():
    pTime = 0
    cTime = 0
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {camera_index}")
        return

    detector = headPose()
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frames")
            continue

        detector.findFace(frame)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

