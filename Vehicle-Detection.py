# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:03:56 2018

@author: henders
"""

import sys
import os
import time
import glob
import pickle

import numpy as np
import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label


#
# Option Flags
#

TRAIN_MODEL = False
CREATE_WRITEUP_IMAGES = False
PROCESS_TEST_IMAGES = False
PROCESS_TEST_FRAMES = False
PROCESS_VIDEO_FILE = True
WRITE_OUTPUT_FRAMES = False


#
# Hyperparameters - used to train the model
#

class Hyperparameters():
    def __init__(self):
        # HOG parameters
        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 48
        self.pix_per_cell = 8
        self.cells_per_block = 2
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        # Bin spatial parameters
        self.spatial_size = (16, 16)
        # Histogram parameters
        self.hist_bins = 32
        # Which features to include?
        self.spatial_features = True
        self.color_hist_features = True
        self.hog_features = True


#
# Window search parameters
#

class SearchParameters():
    def __init__(self):
        # We don't need to look at the entire image.
        # Search only the bottom portion of an image for vehicles.
        self.y_start_stop = [400, 656]
        # Scale the image before searching for objects.
        self.scale = 1.25
        # Search window size
        self.xy_window = (64, 64)
        # Search window overlap (percentage)
        self.xy_overlap = (0.5, 0.5)


#
# Constants
#

# Image text parameters
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_COLOR = (255, 255, 255)
TEXT_THICKNESS = 2
TEXT_LINE_TYPE = cv2.LINE_AA


#
# Spatial Binning Features
#

def bin_spatial(img, size=(32, 32), split_colors=False):
    """
    Compute binned color features.

    Args:
        img: Input image
        size: Resize the image to this size.
        split_colors: Split out color channels or just resize?

    Return:
        A list of features.
    """
    if split_colors:
        color1 = cv2.resize(img[:, :, 0], size).ravel()
        color2 = cv2.resize(img[:, :, 1], size).ravel()
        color3 = cv2.resize(img[:, :, 2], size).ravel()
        return np.hstack((color1, color2, color3))
    else:
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features


#
# Color Histogram Features
#

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    """
    Compute color histogram features.

    Args:
        img: Input image
        nbins (int): Number of histogram bins.
        bins_range: Range of histogram bins.

    Return:
        A list of features.
    """

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features


#
# HOG Features
#

def get_hog_features(img, orient, pix_per_cell, cells_per_block,
                     vis=False, feature_vec=True):
    """
    Compute HOG features and optionally return an image for visualization.

    Args:
        img: Input image.
        orient (int): Number of orientations.
        pix_per_cell (int): Pixels / cell.
        cells_per_block (int): Cells / block for.
        vis: Return an image for visualization?
        feature_vec: Return data as a feature vector (using ravel)?

    Returns:
        A list of features[, an image for visualization]
    """
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm='L2-Hys',
                                  cells_per_block=(cells_per_block, cells_per_block),
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features


#
# Extract Features
#

def extract_features(img, params):
    """
    Extract features from an image.

    Args:
        img: Input image.
        params: Model training parameters.

    Returns:
        A list of all features for all images.
    """

    file_features = []

    # Apply color conversion if necessary.
    if params.color_space in ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']:
        if params.color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif params.color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif params.color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif params.color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif params.color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    # Add binned spatial color features.
    if params.spatial_features:
        spatial_features = bin_spatial(feature_image, size=params.spatial_size)
        file_features.append(spatial_features)

    # Add color histogram features.
    if params.color_hist_features:
        hist_features = color_hist(feature_image, nbins=params.hist_bins)
        file_features.append(hist_features)

    # Add HOG features.
    if params.hog_features:
        if params.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(
                        get_hog_features(feature_image[:, :, channel],
                                         params.orient,
                                         params.pix_per_cell,
                                         params.cells_per_block,
                                         vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, params.hog_channel],
                                            params.orient,
                                            params.pix_per_cell,
                                            params.cells_per_block,
                                            vis=False, feature_vec=True)

        # Append the new feature vector to the features list
        file_features.append(hog_features)

    # Return features (and possibly images)
    return np.concatenate(file_features)


#
# Slide Window
#

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Generate a list of windows for an image.

    Args:
        img: Input image
        x_start_stop: Start / stop position on the x axis (default to image width)
        y_start_stop: Start / stop position on the y axis (default to image height)
        xy_window: Window size for x and y
        xy_overlap: Percentage overlap between windows for the x and y axis.

    Returns:
        A list of windows (bounding boxes).
    """

    image_width, image_height = (img.shape[1], img.shape[0])

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = image_width
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = image_height

    # Compute the span of the region to be searched
    xy_span = [x_start_stop[1] - x_start_stop[0],
               y_start_stop[1] - y_start_stop[0]]

    # Compute the number of pixels per step in x/y
    xy_step = [int(xy_window[0] * xy_overlap[0]),
               int(xy_window[1] * xy_overlap[1])]

    # Compute the number of windows in x/y
    windows_x = int(1 + (xy_span[0] - xy_window[0]) / (xy_window[0] * xy_overlap[0]))  # 18
    windows_y = int(1 + (xy_span[1] - xy_window[1]) / (xy_window[1] * xy_overlap[1]))  # 10
#    total_windows = windows_x * windows_y

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for x_window in range(windows_x):
        for y_window in range(windows_y):
            # Calculate each window position
            x_start = x_start_stop[0] + x_window * xy_step[0]
            x_end = x_start + xy_window[0]
            y_start = y_start_stop[0] + y_window * xy_step[1]
            y_end = y_start + xy_window[1]
            bbox = ((x_start, y_start), (x_end, y_end))

            # Append window position to list
            window_list.append(bbox)

    # Return the list of windows
    return window_list


#
# Draw Boxes
#

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draw bounding boxes on an image.

    Args:
        img: draw on this image
        bboxes: List of bounding boxes.
        color: Box color.
        thick: Box line thickness.

    Returns:
        A copy of the input image with bounding boxes drawn.
    """

    # Make a copy of the image
    imcopy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    # Return the image copy with boxes drawn
    return imcopy


#
# Train Model
#

def train_model(cars, notcars, params):
    """
    Train a SVC model for use in finding vehicles.

    Args:
        cars: A list of vehicle images.
        notcars: A list of non-vehicle images.
        params: Model training params.

    Returns:
        A trained SVC model, A trained StandardScaler for features (X)
    """

    t = time.time()

    car_features = list(map(lambda img_file: extract_features(mpimg.imread(img_file), params), cars))
    notcar_features = list(map(lambda img_file: extract_features(mpimg.imread(img_file), params), notcars))

    t2 = time.time()
    print(round(t2 - t, 2), 'seconds to extract HOG features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler (fit only on training data)
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X (both training data and test data)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using: {} orientations, {} pixels per cell, and {} cells per block'
          .format(params.orient, params.pix_per_cell, params.cells_per_block))
    print('Hog channel: {}'.format(params.hog_channel))
    print('Feature vector length: {}'.format(len(X_train[0])))

    # Use a linear SVC
    svc = LinearSVC()

    # Train the model.
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'seconds to train SVC...')

    # Check the accuracy (score) on the test data.
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample.
    t = time.time()
    n_predict = 10
    print('Prediction test sample size: {}'.format(n_predict))
    print('SVC predict: {}'.format(svc.predict(X_test[0:n_predict])))
    print('Labels:      {}'.format(y_test[0:n_predict]))
    t2 = time.time()
    print(round(t2 - t, 5), 'seconds to predict', n_predict, 'labels with SVC')

    return svc, X_scaler


#
# Find Cars
#

def find_cars(img,
              params,
              svc, X_scaler,
              search_params):
    """
    Find potential car locations in an image.  Extract features using HOG
    subsampling and make predictions as to 'car' or 'not car'.

    Args:
        img: Input image
        params: Model training parameters.
        svc: Trained SVC model for finding vehicles.
        X_scaler: Trained StandardScaler for image features.
        search_params: Window search parameters

    Returns:
        A list of bounding boxes for potential car locations,
        A copy of the input image with bounding boxes drawn
    """

#    # Draw bounding boxes on a copy of the original image.
#    img_detect = np.copy(img)
#
#    bbox_list = []
#    windows = slide_window(img,
#                           y_start_stop=search_params.y_start_stop,
#                           xy_window=search_params.xy_window,
#                           xy_overlap=search_params.xy_overlap)
#    for bbox in windows:
#        img_window = cv2.resize(img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]],
#                                (64, 64))  # Training images are size 64x64
#        features = extract_features(img_window, params)
#
#        scaled_features = X_scaler.transform(features.reshape(1, -1))
#        pred = svc.predict(scaled_features)
#
#        if pred == 1:
#            bbox_list.append(bbox)
#            cv2.rectangle(img_detect, bbox[0], bbox[1], (0, 0, 255), 6)
#
#    return bbox_list, img_detect


    # Draw bounding boxes on a copy of the original image.
    img_detect = np.copy(img)

    #
    # Image pre-processing.
    #

    img = img.astype(np.float32) / 255  # normalize
    img = img[search_params.y_start_stop[0]:search_params.y_start_stop[1], :, :]  # clip

    # Apply color conversion if necessary.
    if params.color_space in ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']:
        if params.color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif params.color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif params.color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif params.color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif params.color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    # Scale
    if search_params.scale != 1:
        imshape = feature_image.shape
        feature_image = cv2.resize(feature_image,
                                   (np.int(imshape[1] / search_params.scale),
                                    np.int(imshape[0] / search_params.scale)))

    #
    # Initialization
    #

    # Since we are using all three channels here for HOG features, we must
    # have set the MODEL_HOG_CHANNEL parameter to 'ALL' else we'll get an
    # error when trying to use the scaler below.
    if params.hog_channel == 'ALL':
        ch1 = feature_image[:, :, 0]
        ch2 = feature_image[:, :, 1]
        ch3 = feature_image[:, :, 2]
    else:
        ch1 = feature_image[:, :, params.hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // params.pix_per_cell) - params.cells_per_block + 1
    nyblocks = (ch1.shape[0] // params.pix_per_cell) - params.cells_per_block + 1
#    nfeat_per_block = orient * cells_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // params.pix_per_cell) - params.cells_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image here so
    # we need only do it once.
    hog1 = get_hog_features(ch1, params.orient, params.pix_per_cell, params.cells_per_block, feature_vec=False)
    if params.hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, params.orient, params.pix_per_cell, params.cells_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, params.orient, params.pix_per_cell, params.cells_per_block, feature_vec=False)

    #
    # Find cars
    #

    bbox_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * params.pix_per_cell
            ytop = ypos * params.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(feature_image[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=params.spatial_size, split_colors=True)
            hist_features = color_hist(subimg, nbins=params.hist_bins)

            # Scale features and make a prediction
            combined_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(combined_features)
            test_prediction = svc.predict(test_features)

            # If the model indicates the presence of a car, add the bounding
            # box to our list and draw it on the return image.
            if test_prediction == 1:
                xbox_left = np.int(xleft * search_params.scale)
                ytop_draw = np.int(ytop * search_params.scale)
                win_draw = np.int(window * search_params.scale)
                bbox = ((xbox_left, ytop_draw + search_params.y_start_stop[0]),
                        (xbox_left + win_draw, ytop_draw + win_draw + search_params.y_start_stop[0]))
                bbox_list.append(bbox)
                cv2.rectangle(img_detect, bbox[0], bbox[1], (0, 0, 255), 6)

    return bbox_list, img_detect


#
# Add Heat
#

def add_heat(heatmap, bbox_list):
    """
    Given a heatmap, add one to each pixel inside each bounding box in the
    specified list.

    Args:
        heatmap: Modify and return this heatmap.
        bbox_list: List of bounding boxes for potential car locations.

    Returns:
        A modified heatmap
    """

    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


#
# Apply Threshold
#

def apply_threshold(heatmap, threshold):
    """
    Apply a threshold to a heatmap. Zero out pixels that do not meet
    the threshold.

    Args:
        heatmap: Modify and return this heatmap.
        threshold (int): Apply this threshold.

    Returns:
        A modified heatmap
    """

    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    # Return thresholded map
    return heatmap


#
# Draw Labeled Boxes
#

def draw_labeled_bboxes(img, labels, color=(0, 0, 255), thick=6):
    """
    Draw labeled bounding boxes on an image.

    Args:
        img: Modify and return this image.
        labels: Labeled bonding boxes.
            See scipy.ndimage.measurements import label
        color: Box color.
        thick: Box line thickness.

    Returns:
        A modified image
    """

    # For each detected car ...
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value.
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels.
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y.
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)

    # Return the image
    return img


#
# Detect vehicles in a video clip
#

class VehicleDetection():
    """
    Define a class to encapsulate video processing for vehicle detection.
    """

    def __init__(self, pickle_file, params, search_params):
        # Parameters
        self.params = None
        # Model
        self.model = None
        # StandardScaler for features
        self.scaler = None
        # Window search parameters
        self.search_params = None
        # Current frame number.
        self.current_frame = 0
        # Output dir for modified video frames.
        self.video_dir = None

        self.params = params
        self.search_params = search_params

        if TRAIN_MODEL or not os.path.isfile(pickle_file):
            print('Training model ...')

            # Get train / test data filenames
            cars = glob.glob('./train_test_data/vehicles/*/*.png', recursive=True)
            print('Found {} images of vehicles.'.format(len(cars)))
            notcars = glob.glob('./train_test_data/non-vehicles/*/*.png', recursive=True)
            print('Found {} images of non-vehicles.'.format(len(notcars)))

            # Train the model
            self.model, self.scaler = train_model(cars, notcars, self.params)

            # Write trained model to a pickle file.
            pickle_data = {
                'svc': self.model,
                'scaler': self.scaler
            }
            with open(pickle_file, 'wb') as f:
                # Pickle the 'pickle_data' dictionary using the highest
                # protocol available.
                pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

        else:
            # Load from pickle file
            print('Loading model ...')
            with open(pickle_file, 'rb') as f:
                # The protocol version used is detected automatically, so we
                # do not have to specify it.
                pickle_data = pickle.load(f)

            # Unpack attributes from pickle data dictionary.
            self.model = pickle_data['svc']
            self.scaler = pickle_data['scaler']
#            orient = pickle_data['orient']
#            pix_per_cell = pickle_data['pix_per_cell']
#            cells_per_block = pickle_data['cells_per_block']
#            spatial_size = pickle_data['spatial_size']
#            hist_bins = pickle_data['hist_bins']

    def ProcessVideoClip(self, input_file, video_dir=None):
        """
        Apply the FindVehiclesVideoFrame() function to each frame in a given
        video file. Save the results to a new video file in the same location
        using the same filename but with "_vehicles" appended.

        Args:
            input_file (str): Process this video file.
            video_dir (str): Optional location for modified video frames.

        Returns:
            none

        To speed up the testing process or for debugging we can use a subclip
        of the video. To do so add

            .subclip(start_second, end_second)

        to the end of the line below, where start_second and end_second are
        integer values representing the start and end of the subclip.
        """
        self.video_dir = video_dir

        # Open the video file.
        input_clip = VideoFileClip(input_file)  # .subclip(40, 45)

        # For each frame in the video clip, replace the frame image with the
        # result of applying the 'FindLaneLines' function.
        # NOTE: this function expects color images!!
        self.current_frame = 0
        output_clip = input_clip.fl(self.FindVehiclesVideoFrame)

        # Save the resulting, modified, video clip to a file.
        file_name, ext = os.path.splitext(input_file)
        output_file = file_name + '_vehicles' + ext
        output_clip.write_videofile(output_file, audio=False)

        # Cleanup
        input_clip.reader.close()
        input_clip.audio.reader.close_proc()
        del input_clip
        output_clip.reader.close()
        output_clip.audio.reader.close_proc()
        del output_clip

    def FindVehiclesVideoFrame(self, get_frame, t):
        """
        Given an image (video frame) find vehicles. Draw bounding boxes
        around each detected vehicle on a copy of the input image and
        return the result.

        Args:
            get_frame: Video clip's get_frame method.
            t: time in seconds

        Returns:
            modified copy of the input image
        """
        self.current_frame += 1

        img = get_frame(t)  # RGB
#        img_size = (img.shape[1], img.shape[0])

        # Find vehicles.
        img_detect = self.FindVehicles(img)

        # Write the frame number to the image.
        frame = 'Frame: {}'.format(self.current_frame)
        cv2.putText(img_detect, frame, (1050, 30),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        # Write the time (parameter t) to the image.
        time = 'Time: {}'.format(int(round(t)))
        cv2.putText(img_detect, time, (1050, 700),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, TEXT_LINE_TYPE)

        # Optionally write the modified image to a file.
        if self.video_dir is not None:
            output_file = os.path.join(self.video_dir,
                                       'frame{:06d}.jpg'.format(self.current_frame))
            mpimg.imsave(output_file, img_detect)

        # Return the modified image.
        return img_detect

    def ProcessTestImages(self, image_dir, output_dir):
        """
        Process images in the test directory and create output images showing
        the results of various stages of the image processing pipeline.

        Args:
            image_dir (str): read test images from this directory
            output_dir (str): write modified images to this directory

        Returns:
            nothing
        """
        images = glob.glob(os.path.join(image_dir, '*.jpg'))
        for fname in images:
            print('Processing image {}'.format(fname))
            _, name = os.path.split(fname)
            name, ext = os.path.splitext(name)

            # Read the image.
            img = mpimg.imread(fname)  # RGB

            # Find vehicles
            self.FindVehicles(img, output_dir=output_dir, img_name=(name, ext))

    def CreateWriteupImages(self, image_list, output_dir, output_label):
        """
        Create sample images for use in the project writup. Randomly choose
        a single image file from the given list.

        Args:
            image_list [str]: list of image file names
            output_dir (str): write modified images to this directory
            output_label (str): label for output file (e.g., car)

        Returns:
            nothing
        """
        i = np.random.randint(0, len(image_list))
        fname = image_list[i]

        print('Processing image {}'.format(fname))
        _, name = os.path.split(fname)
        name, ext = os.path.splitext(name)

        # Read the image.
        img = mpimg.imread(fname)  # RGB

        # Save a copy of the original image to the output directory.
        mpimg.imsave(os.path.join(output_dir, output_label) + ext,
                     img)

        # Get hog visualization images
        if self.params.hog_channel == 'ALL':
            for channel in range(img.shape[2]):
                _, hog_img = get_hog_features(img[:, :, channel],
                                              self.params.orient,
                                              self.params.pix_per_cell,
                                              self.params.cells_per_block,
                                              vis=True, feature_vec=True)
                mpimg.imsave(os.path.join(output_dir, output_label + '_1_hog_' + str(channel + 1)) + ext,
                             hog_img)
        else:
            _, hog_img = get_hog_features(img[:, :, self.params.hog_channel],
                                          self.params.orient,
                                          self.params.pix_per_cell,
                                          self.params.cells_per_block,
                                          vis=True, feature_vec=True)
            mpimg.imsave(os.path.join(output_dir, output_label + '_1_hog_' + str(self.params.hog_channel)) + ext,
                         hog_img)

    def FindVehicles(self, img, output_dir=None, img_name=(None, None)):
        """
        Find vehicles in a single image. Assume the image is RGB.

        Args:
            img: input image
            output_dir (str): location for output image files
            img_name (str, str): image file name and extension
            img_ext (str): image file extension

        Returns:
            a modified image with vehicles marked with bounding boxes
        """

        vis = output_dir is not None

        #
        # Find potential vehicle locations
        #

        box_list, box_img = find_cars(img,
                                      self.params, self.model, self.scaler,
                                      self.search_params)
        if vis:
            mpimg.imsave(os.path.join(output_dir, img_name[0] + '_1_bboxes') + img_name[1],
                         box_img)

        #
        # Heat map - remove false positive cars
        #

        # Add heat to each box in box_list.
        heat = np.zeros_like(box_img[:, :, 0]).astype(np.float)
        heat = add_heat(heat, box_list)

        # Apply threshold to help remove false positives.
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying.
        heatmap = np.clip(heat, 0, 255)
        if vis:
            mpimg.imsave(os.path.join(output_dir, img_name[0] + '_2_heatmap') + img_name[1],
                         heatmap, cmap='hot')

        # Find final boxes from heatmap using label function.
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        if vis:
            mpimg.imsave(os.path.join(output_dir, img_name[0] + '_3_cars') + img_name[1],
                         draw_img)

        return draw_img


#
# Main
#

def main(name):

    print('Name: {}'.format(name))
    print()

    params = Hyperparameters()
    search_params = SearchParameters()
    proc = VehicleDetection('./svc_pickle.p', params, search_params)

    if CREATE_WRITEUP_IMAGES:
        # Get train / test data filenames
        cars = glob.glob('./train_test_data/vehicles/*/*.png', recursive=True)
        notcars = glob.glob('./train_test_data/non-vehicles/*/*.png', recursive=True)
        # Create images for the project writeup
        proc.CreateWriteupImages(cars, './output_images', 'car')
        proc.CreateWriteupImages(notcars, './output_images', 'notcar')

    if PROCESS_TEST_IMAGES:
        proc.ProcessTestImages('./test_images', './output_images')
    if PROCESS_TEST_FRAMES:
        proc.ProcessTestImages('./test_frames', './test_frames_output')
    if PROCESS_VIDEO_FILE:
        video_dir = None
        if WRITE_OUTPUT_FRAMES:
            video_dir = './project_video_lanes'
        proc.ProcessVideoClip('./project_video.mp4', video_dir)


if __name__ == '__main__':
    main(*sys.argv)
