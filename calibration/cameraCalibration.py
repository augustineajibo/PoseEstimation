import numpy as np
import cv2 as cv
import glob
import os

#### Find chessboard corners - object point and image point #####
chessboardSize = (23, 16)  # Ensure this matches your actual printed pattern!
frameSize = (1440, 1080)

# Prepare object points
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Arrays to store object and image points
objPoints = []  # 3D points in real world
imgPoints = []  # 2D points in image plane

images = glob.glob("captured_images/*.jpg")

if not images:
    print("❌ No images found in 'captured_images/' directory.")
    exit()

print(f"🔍 Found {len(images)} image(s) to process...")

for image_path in images:
    print(f"Processing {image_path}")
    img = cv.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)

        # Optional: Visual feedback
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow("Detected Corners", img)
        cv.waitKey(500)
    else:
        print(f"❌ Chessboard not detected in {image_path}")

cv.destroyAllWindows()

### Calibration ###
if len(objPoints) == 0 or len(imgPoints) == 0:
    print("❌ No valid chessboard corners found. Calibration aborted.")
    exit()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)
print("✅ Camera Calibrated:", ret)
print("Camera Matrix:\n", cameraMatrix)
print("Distortion Coefficients:\n", dist)

# Load test image
img = cv.imread("captured_images/cali5.jpg")
if img is None:
    print("❌ Calibration image 'cali5.jpg' not found.")
    exit()

h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite("caliResult1.jpg", dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
dst = dst[y:y+h, x:x+w]
cv.imwrite("caliResult2.jpg", dst)

# Reprojection Error
mean_error = 0
for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2) / len(imgPoints2)
    mean_error += error

print(f"\n📏 Total reprojection error: {mean_error / len(objPoints):.4f}")
