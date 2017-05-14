## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Report_imgs/camera_cali.png "Undistorted"
[image2]: ./Report_imgs/camera_cali_test.png "Road Transformed"
[image3]: ./Report_imgs/color_viz.png "Binary Example"
[image4]: ./Report_imgs/binary_output.png "Warp Example"
[image5]: ./Report_imgs/pers_transformed.png "Fit Visual"
[image6]: ./Report_imgs/final.png "Output"
[image7]: ./examples/color_fit_lines.jpg "Output"
[image8]: ./Report_imgs/sliding_window.png "Output"
[image9]: ./Report_imgs/margin_search.png "Output"
[video1]: ./project_video.mp4 "Video"

---
### Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients to un-distort video from the camera.

The code for this step is contained in the code cell titled "2) Camera Calibration" of the jupyter notebook named Land_Finding_Pipeline.ipynb .

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

The find cv2.findChessboardCorners did not work for all the calibration images as some of the images edges are out of view. But we were still able to calibrate the camera with the remaining images.

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]


#### 2. Perpective Transform

The code for my perspective transform can be found in the function `perspective_transform()`, which appears in under the header " 3) Perpective Transform" in the file `Lane_Finding_Pipeline.ipynb` .  The function takes as inputs an image (`img`) and returns a transformed view similar to a birds-eye view

```python
def perspective_transform(img):
    img_size = (img.shape[1], img.shape[0]
    src = np.float32([[190, 720], [582, 457], [701, 457], [1145, 720]])
    offset = [150,0]
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, np.array([src[3, 0], 0]) - offset, src[3] - offset])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image,matrix and inv matrix
    return warped, M, Minv
```

An example of the result of this function


![alt text][image5]

#### 3. Color Thresholding

I started by exploring the various color spaces and edge thresholding techniques. Just by using the B channel of LAB color space for yellow lines and L channel of HLS colorspace for white lines I was able to get very clean and distinctive lane lines.

![alt text][image3]

In order to make the color thresholding more robust, a different set of thresholds was applied for varying road conditions to get optimal results. For situations when the road was faded or the road has overcast shadows. The condition of the road is estimated using the mean of the Y channel of the YUV colorspace.

![alt text][image4]

Code for the color processing functions can be found in the jupyter notebook under the title "4) Color and Edge Thresholding"



#### 4. Fitting polynomial curves to the detected lane lines

![alt text][image7]

Once we have our binary image of the lane lines, we take the bottom half of the image and plot its points on a histogram. We can then find the peaks of the histogram, which signifies that most of the points are in that region. So we place our first window for both the left and right lanes and search for the lane pixels in that window. Then we slide the window to the next section and search, till we reach the end of the images.

![alt text][image8]

Once the first lane has been detected we can search within a certain margin of that line in the subsequent frames reducing the computational complexity.

![alt text][image9]

##### Lane validation
In order to ensure accurate results and increase the robustness of the detection we have to validate the lanes with a set of rules to ensure that what we have detected are indeed accurate lane lines.


We do this using these rules:
* Checking if the lane width is too small or large
* Checking if the left and right detected lane are on their respective side of the image
* Checking that the lane has a minimum of n points

If the lane cannot be validated the lane reading will not be included and will revert back to a sliding window search to find a more accurate lane line.




#### 5. Radius of Curvature Calculation and Vehicle offset calculation

Radius of curvature calculation is done in the last part of the  update_lane function in the Line Class

```python
self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
```
Vehicle offset calculation is done in the last part of the  lane_validation_and_update function

```python
lane_center = right_line.bestx - left_line.bestx
left_line.line_base_pos = ((img.shape[1]*0.5 - lane_center)*3.7)/700
right_line.line_base_pos = left_line.line_base_pos
```



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `draw_lane(undist, img, Minv)` .  Here is an example of my result on a test image:

I included the other steps of the pipeline in the video to aid in the process of debugging and finetuning the solution. This was implemented in the build_frame function.

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The robustness of the solution relies heavily on the choice of colorspace and the validation logic to ensure that only the correct lane detections are considered in final calculations. Initially I was trying to utilize a single set of thresholds to robustly identify the lanes in changing condition. But just like how our eyes and cameras work differently under different conditions, varying conditions called for varying set of thresholds.Thresholds that dynamically change based on road conditions. However this is not the ideal way, as one would have to manually add all the various scenarios and conditions for different type of roads and weathers for a truly robust solution. Ideally we would want to use deep learning for a pixel segmentation(SegNet). To train it to recognize more scenarios we would simply have to add more varied and diverse data and the model would generalize a visual identification filter for lane lines.
