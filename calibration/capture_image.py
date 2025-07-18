import cv2
import os
from datetime import datetime

def capture_image_from_camera(camera_index=1, save_dir="captured_images"):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Open the external camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot access camera with index {camera_index}")
        return

    print("Press SPACE to capture image, or ESC to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        cv2.imshow("External Camera - Press SPACE to capture", frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC key
            print("Escape hit, closing...")
            break
        elif key % 256 == 32:  # SPACE key
            # Generate filename with timestamp
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Image saved: {filepath}")

    cap.release()
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    capture_image_from_camera(camera_index=4)  # Change index if needed
