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
3. The object points are the known coordinates of a 9x6 chessboard in 3D (x,y,z) coordinates with z = 0 for all points
4. Use the cv2.findChessboardCorners() method to find the image points.  If the function detects the 9x6 chessboard corners, the image will be used to calibrate the camera.
5. Pass the detected image points and objects points for all calibration images to the cv2.calibrateCamera() method to calibrate the camera

![ScreenShot](image1.png)

```
nx = 9   
ny = 6

objpoints = []
imgpoints = []

# prepare object point [0,0,0], [1,0,0], [2,0,0]...[8,5,0]
objp = np.zeros((nx * ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# loop through the calibration images to find the objpoints and imgpoints
images = glob.glob("camera_cal/calibration*.jpg")
for filename in images:
    img = mpimg.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # only include those that meet the nx, ny requirement 
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

# find parameters to calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

```

### Software pipeline to identify the lane boundaries 
Now that I calibrated a camera and found the perspective transform matrix to transform the road to bird's eye view, it is ready to create a software pipeline to identify the lane boundaries.


#### Step 2: Apply a distortion correction to raw images

1. Used cv2.undistort() method to correct raw images from the camera. 

![ScreenShot](image2.png)

#### Step 3: Use color transforms, gradients, etc., to create a thresholded binary image

The code for this step is contained in the 6th & 7th code cell of the IPython notebook located in "./advanced_lane_lines_for_submission.ipynb".

1. I converted the undistorted image to HLS color map
2. Next, I used the S channel to find the color thresholded binary image 

    ```
    s_channel_binary = s_channel_thresh(undist, thresh=(150, 255))    
    ```
3. Next, I used L channel to find the sobel gradient x and sobel gradient y thresholded binary image

    ```    
    gradx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    ```
4. Used L channel to find the sobel magnitude and direction thresholded binary image 

    ```
    mag_binary = mag_thresh(l_channel, sobel_kernel=ksize, mag_thresh=(40, 100))
    dir_binary = dir_threshold(l_channel, sobel_kernel=ksize, thresh=(0.7, 1.2))
    ```

5. Combined the color, sobel gradients and direction to create the final thresholded binary image for lane detection 

    ```
    combined_binary = np.zeros_like(s_channel_binary)
    combined_binary[(s_channel_binary == 1) | (gradx == 1) & (grady == 1) | (mag_binary == 1) & (dir_binary == 1)] = 1
    ```

6. I also set all pixels from 700 to 719 to 0 because that is the front of car and not part of the road to avoid sun reflections on the front of the car affects the lane detection.

    ```
    combined_binary[700:,:] = 0 
    ```

![ScreenShot](image5.png)


#### Step 4: Apply a perspective transform to rectify binary image ("birds-eye view")

1. Next, I found the perspective transform matrix to create the birds-eye view of the road.  I drew red lines on the undistorted image and then tested different src/dst combinations to make sure the red lines were parallel after the transformation. 

```
s1 = (230, 700)
s2 = (595, 450)
s3 = (685, 450)
s4 = (1082, 700)

d1 = (200, 700)
d2 = (200, 0)
d3 = (900, 0)
d4 = (900, 700)

src = np.array([s1, s2, s3, s4], np.float32)
dst = np.array([d1, d2, d3, d4], np.float32)

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 230, 700      | 200, 700      | 
| 595, 450      | 200, 0        |
| 685, 450      | 900, 0        |
| 1082, 700     | 900, 700      |

![ScreenShot](image4.png)







