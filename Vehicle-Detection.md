# **Vehicle Detection** 

## Scott Henderson
## Self Driving Car Nanodegree Term 1 Project 5

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png "Car"
[image2]: ./output_images/car_1_hog_1.png "Car HOG Channel 1"
[image3]: ./output_images/car_1_hog_2.png "Car HOG Channel 2"
[image4]: ./output_images/car_1_hog_3.png "Car HOG Channel 3"
[image5]: ./output_images/notcar.png "Not Car"
[image6]: ./output_images/notcar_1_hog_1.png "Not Car HOG Channel 1"
[image7]: ./output_images/notcar_1_hog_2.png "Not Car HOG Channel 2"
[image8]: ./output_images/notcar_1_hog_3.png "Not Car HOG Channel 3"

[image9]: ./test_images/test_image.jpg "Original"
[image10]: ./output_images/test_image_1_bboxes.jpg "Bounding Boxes"
[image11]: ./output_images/test_image_2_heatmap.jpg "Heatmap"
[image12]: ./output_images/test_image_3_cars.jpg "Cars"

[video1]: ./project_video.mp4 "Video"
[video2]: ./project_video_vehicles.mp4 "Video With Cars"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

All code for this project is in the Python script file "./Vehicle-Detection.py".  All line numbers below refer to this source file.

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features were extracted using the `get_hog_features()` function, lines 139-174. This is really just a call to the `skimage.feature.hog` function with parameters set by trial and error. This is certainly one area for improvement. See notes at the end of this writeup.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image5]

After settling on HOG parameters (see the next section) I produced images showing the results of running the HOG algorithm for each class of image:

Car:
![alt text][image2]
![alt text][image3]
![alt text][image4]

Not car:
![alt text][image6]
![alt text][image7]
![alt text][image8]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various values more or less at random for the color space, # of orientations and # of pixels per cell, and # of cells per block parameters. I left the hog channels parameter at 'ALL' because this seemed like an easier approach given that I was also changing the color space parameter. Again, I think a more rigorous approach to parameter finding is called for (and again see note below).

In order to make it easier to pass parameters around and to ensure consistency between parameters used for model training and for prediction with the trained model, I created a Python class to hold the final parameter values, lines 42-57.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used a linear SVM model for this project since that was the model used in the class demos. Training consisted of reading all vehicle and non-vehicle images from the training data set and then creating labels for each. A train/test split was made reserving 20% of the data for testing. A StandardScaler object was fitted on the training features only and then used to scale both the training features and the test features. After fitting the model the resulting accuracy on the test set was good - varying between 98% and 99%.

Since I did not see this sort of accuracy in my processed test images or video frames I conclude that there are issues with my implementation of the window search. Yet another area for improvement.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried a couple of different approaches to implementin a sliding window search. I started with the sample code in the `find_cars()` function given in the class lectures. This worked for the most part but was not particularly satisfactory. I then attempted to use the `search_windows()` function from one of the class quizes but this proved even less satisfactory. In the end my implementation is basically the `find_cars()` function from the class lecture, lines 415-461.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I ran my final code pipeline on the images in the test image folder plus this image taken from the class lecture notes:

![alt text][image9]

Here is the result of the sliding window search on the test image with potential car bounding boxes drawn:

![alt text][image10]

In order to remove false positive bounding boxes I essentially used the code from the class lecture, lines 574-662.

The resulting heatmap for the test image:

![alt text][image10]

And the final result for the test image showing the locations of positively identified cars:

![alt text][image11]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_vehicles.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

As mentioned above this is just the code from the class lecture, lines lines 574-662.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach was to start with the sample code from the online lessons. After that I really just added a bit of code for processing test images and video files (taken from my previous Advanced Lane Lines project) and then spent the bulk of the remainder of my time trying different parameter values to see which worked best.

One area that is crying out for improvement is a more systematic approach to finding parameter values. For example, the HOG feature extraction process uses several parameters: color space, # of orientations, @ of pixels per cell, # of cells per block, # of channels (one of three or 'ALL'). Any sort of non-automated systematic search is not practical for these parameters. Perhaps a neural network or some other automated parameter serach algorithm would be helpful.  There are also other parameters such as the spatial size used for color binning and the # of histogram bins used for color histogram generation. It is not clear to me what sort of impact any of these values may have on the accuracy of the ultimate "car" vs "not car" results and it is likely that the only way to find the best parameters is to try a large selection of different combinations. Which brings us back to using some sort of automated approach.

Another area for further investigation is to try other models. Linear SVM seems to work fairly well but perhaps a different model, Decision Tree, etc., would be better. Or even an ensemble of several models.

Also the perceived accuracy of the process on the test images and the video frames (via visual inspection only, I did not take the time to manually tag the actual vehicles in the test images) did not approach the ~98% value I saw after training the model. This leads me to believe that there may be issues with my implementation of the sliding window approach to breaking up the input image into chunks that can be fed into the model. Again more investigation is called for here.


