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

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from scipy.ndimage.measurements import label

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip


#
# Option Flags
#

TRAIN_MODEL = True


#
# Hyperparameters
#

#MODEL_COLOR_SPACE = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#MODEL_ORIENT = 6
#MODEL_PIX_PER_CELL = 4
#MODEL_CELLS_PER_BLOCK = 2
#MODEL_HOG_CHANNEL = 'ALL'  # Can be 0, 1, 2, or "ALL"
#
## binning paremeters
#MODEL_SPATIAL_SIZE = 32
#MODEL_HIST_BINS = 64


MODEL_COLOR_SPACE = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
MODEL_ORIENT = 18
MODEL_PIX_PER_CELL = 6
MODEL_CELLS_PER_BLOCK = 2
MODEL_HOG_CHANNEL = 'ALL'  # Can be 0, 1, 2, or "ALL"

# binning paremeters
MODEL_SPATIAL_SIZE = 64
MODEL_HIST_BINS = 64


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cells_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm='L2-Hys',
                                  cells_per_block=(cells_per_block, cells_per_block),
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32), split_colors=False):
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


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs,
                     color_space='RGB',
                     spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9,
                     pix_per_cell=8,
                     cells_per_block=2,
                     hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                        orient, pix_per_cell, cells_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cells_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy



def train_model(cars, notcars):
    t = time.time()
    car_features = extract_features(cars,
                                    color_space=MODEL_COLOR_SPACE,
                                    spatial_size=(MODEL_SPATIAL_SIZE, MODEL_SPATIAL_SIZE),
                                    hist_bins=MODEL_HIST_BINS,
                                    orient=MODEL_ORIENT,
                                    pix_per_cell=MODEL_PIX_PER_CELL,
                                    cells_per_block=MODEL_CELLS_PER_BLOCK,
                                    hog_channel=MODEL_HOG_CHANNEL)
    notcar_features = extract_features(notcars,
                                       color_space=MODEL_COLOR_SPACE,
                                       spatial_size=(MODEL_SPATIAL_SIZE, MODEL_SPATIAL_SIZE),
                                       hist_bins=MODEL_HIST_BINS,
                                       orient=MODEL_ORIENT,
                                       pix_per_cell=MODEL_PIX_PER_CELL,
                                       cells_per_block=MODEL_CELLS_PER_BLOCK,
                                       hog_channel=MODEL_HOG_CHANNEL)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')

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

    print('Using: ', MODEL_ORIENT, ' orientations ', MODEL_PIX_PER_CELL,
          'pixels per cell and ', MODEL_CELLS_PER_BLOCK, ' cells per block')
    print('Hog channel: {}'.format(MODEL_HOG_CHANNEL))
    print('Feature vector length: ', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    return svc, X_scaler


# Define a single function that can extract features using hog sub-sampling
# and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
              cells_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
#    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YCrCb)
#    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1] / scale),
                                      np.int(imshape[0] / scale)))

    # Since we are using all three channels here for HOG features, we must
    # have set the MODEL_HOG_CHANNEL parameter to 'ALL' else we'll get an
    # error when trying to use the scaler below.
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cells_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cells_per_block + 1
#    nfeat_per_block = orient * cells_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cells_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cells_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cells_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cells_per_block, feature_vec=False)

    box_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size, split_colors=True)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            combined_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(combined_features)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                box_list.append(((xleft, ytop + ystart),
                                 (xleft + window, ytop + ystart + window)))
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img,
                              (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                              (0, 0, 255), 6)

    return box_list, draw_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


#
# Main
#

def main(name):

    print('Name: {}'.format(name))
    print()

    svc_pickle_file = './svc_pickle.p'
    if TRAIN_MODEL or not os.path.isfile(svc_pickle_file):
        print('Training model ...')

        cars = glob.glob('./train_test_data/vehicles/*/*.png', recursive=True)
        print('Found {} images of vehicles.'.format(len(cars)))
        notcars = glob.glob('./train_test_data/non-vehicles/*/*.png', recursive=True)
        print('Found {} images of non-vehicles.'.format(len(notcars)))

        # Train the model
        svc, X_scaler = train_model(cars, notcars)

        # Write trained model to a pickle file.
        pickle_data = {
            'svc': svc,
            'scaler': X_scaler
        }
        with open(svc_pickle_file, 'wb') as f:
            # Pickle the 'pickle_data' dictionary using the highest protocol
            # available.
            pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        # Load from pickle file
        print('Loading model ...')

        with open(svc_pickle_file, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            pickle_data = pickle.load(f)

        # Unpack attributes from pickle data dictionary.
        svc = pickle_data['svc']
        X_scaler = pickle_data['scaler']
#        orient = pickle_data['orient']
#        pix_per_cell = pickle_data['pix_per_cell']
#        cell_per_block = pickle_data['cell_per_block']
#        spatial_size = pickle_data['spatial_size']
#        hist_bins = pickle_data['hist_bins']


    #
    # Run the model on a test image.
    #

    img = mpimg.imread('./test_images/test_image.jpg')  # RGB

    # We don't need to look at the entire image.
    ystart = 400
    ystop = 656
    # Scale the image before searching for objects.
    scale = 1.5

    box_list, out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                                  MODEL_ORIENT, MODEL_PIX_PER_CELL,
                                  MODEL_CELLS_PER_BLOCK,
                                  (MODEL_SPATIAL_SIZE, MODEL_SPATIAL_SIZE),
                                  MODEL_HIST_BINS)
    plt.imshow(out_img)


    #
    # Heat map
    #

    # Add heat to each box in box list
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    fig = plt.figure(figsize=(12, 12))
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()


if __name__ == '__main__':
    main(*sys.argv)
