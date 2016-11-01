"""image_processing.py by Mahdi Ramadan, 07-12-2016
This program will be used for video/image processing of
behavior video
"""
import os
from excel_processing import ExcelProcessing as ep
import numpy as np
import cv2
import pandas
import time
import tables
import h5py
from sklearn import preprocessing
from multiprocessing import Process
import warnings
from math import isnan
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
from skimage import data, color, exposure
from skimage.transform import pyramid_gaussian
from skimage.feature import hog



def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_whole_video(exp_folder, lims_ID):
    #initializes video pointer for video of interest based on lims ID
    file_string = get_file_string(exp_folder + '\Videos\\', lims_ID)
    video_pointer = cv2.VideoCapture(file_string)

    # import wheel data
    dir = os.path.join(exp_folder + '\Wheel\\', 'dxds' + str(lims_ID) + '.pkl' )
    wheel = joblib.load(dir)
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]
    k = first_index[0]
    imp = Imputer(missing_values='NaN', strategy='mean')
    wheel = imp.fit_transform(wheel)
    wheel = preprocessing.MinMaxScaler((-1, 1)).fit(wheel).transform(wheel)

    video_pointer.set(1, k)
    ret, frame = video_pointer.read()

    # crops and converts frame into desired format
    frame = cv2.cvtColor(frame[160:400, 100:640], cv2.COLOR_BGR2GRAY)

    prvs = frame
    nex = frame

    # initialize vectors to keep track of data
    count = 1
    mod = 0
    opticals = []
    angles = []
    frames = []

    # length of movie
    limit = int(video_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fps = video_pointer.get(cv2.cv.CV_CAP_PROP_FPS)


    # create hdf file
    hf = h5py.File('data_' + str(lims_ID) + 'test.h5', 'w')
    g = hf.create_group('feature space')
    vector = np.zeros((limit, 4321))
    table = g.create_dataset('features', data=vector, shape=(limit, 4321))

    while count < limit-k:
        prvs = nex
        frames = process_input(prvs)

        ret, frame = video_pointer.read()
        nex = cv2.cvtColor(frame[160:400, 100:640], cv2.COLOR_BGR2GRAY)

        optical = optical_flow(prvs, nex)
        opticals = optical['mag']
        angles= optical['ang']
        vector_data = np.concatenate((np.reshape(wheel[k], (1)), frames, opticals, angles))
        table[count, :] = vector_data

        count += 1
        k += 1

        if count%1000 == 0:
            print (count)

def optical_flow(prvs, next):

    # calculate optical flow and angle of out two frame
    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang= cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # get histograms of optical flow and angles (these are our features)
    mag = process_input(mag)
    ang = process_input((ang*180/np.pi/2))

    return {'mag': mag, 'ang': ang}

def process_input(input):

    frame_data = []


    for (i, resized) in enumerate(pyramid_gaussian(input, downscale=1.5)):
        # if the image is too small, break from the loop


        # winSize = (32,32)
        # blockSize = (16, 16)
        # blockStride = (8, 8)
        # cellSize = (8, 8)
        # nbins = 9
        # derivAperture = 1
        # winSigma = 4.
        # histogramNormType = 0
        # L2HysThreshold = 2.0000000000000001e-01
        # gammaCorrection = 0
        # nlevels = 64
        # hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
        #                         histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        # # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        # winStride = (8, 8)
        # padding = (8, 8)
        #
        # image = color.rgb2gray(resized)
        #
        # hist = hog.compute(image, winStride, padding)
        fd = hog(resized, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=False)


        hist= preprocessing.MinMaxScaler((-1, 1)).fit(fd).transform(fd)
        frame_data = np.concatenate((frame_data, hist))
        
        if resized.shape[0] < 100 or resized.shape[1] < 100:
            break


    # for each defined window over the data, bin values and return a
    # properly scaled list of expectation values
    for (x, y, window) in sliding_window(input, 30, (30, 30)):
        hist, bin = np.histogram(window, 10)
        center = (bin[:-1] + bin[1:]) / 2
        hist_x = np.multiply(center, hist)
        hist_x = preprocessing.MinMaxScaler((-1, 1)).fit(hist_x).transform(hist_x)
        frame_data = np.concatenate((frame_data, hist_x))
    return frame_data

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def get_file_string(exp_folder,lims_ID):

    for file in os.listdir(exp_folder):
        if file.endswith(".avi") and file.startswith(lims_ID):
            file_string = os.path.join(exp_folder, file)
            return file_string



if __name__ == '__main__':
    # initializes code with path to folder and lims ID
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    exp_folder = 'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation'
    lims_ID = '501560436'
    # run_whole_video(exp_folder, lims_ID)

    p = Process(target=run_whole_video(exp_folder, lims_ID), args= (exp_folder, lims_ID))
    p.start()
    p.join()