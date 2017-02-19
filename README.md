# CarND-Advanced-Lane-Lines

## Advanced Lane Finding Project

### Goal
The goal of this project is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. 

### Steps to complete this project are the following:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

#### Step 1: Computer the camera calibration matrix and distortion coefficients given a set of chessboard images

The code for this step is contained in the 4th code cell of the IPython notebook located in "./advanced_lane_lines_for_submission.ipynb".

1. First, Read in calibration images of the chessboard.  it is recommended to use at least 20 images to calibrate the camera
2. Map the corner coodinates of the 2D images (image points) to the 3D coordiates of the real undistorted chessboard corners (object points)
3. The object points are the known coordinates of a 9x6 chessboard in 3D coordinates with z = 0 for all points
4. Use the cv2.findChessboardCorners() method to find the image points.  If the function detects the 9x6 chessboard corners, the image will be used to calibrate the camera.
5. Pass the image points and objects points to the cv2.calibrateCamera() method to calibrate the camera

![ScreenShot](image1.png)
