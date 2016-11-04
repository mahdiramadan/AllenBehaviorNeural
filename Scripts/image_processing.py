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
from excel_processing import ExcelProcessing as ep



def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_whole_video(exp_folder, lims_ID):
    #initializes video pointer for video of interest based on lims ID
    file_string = get_file_string(exp_folder + '\Videos\\', lims_ID)
    video_pointer = cv2.VideoCapture(file_string)

    # load feature scalers
    scale_wheel= joblib.load(exp_folder + '\Scalers\\' + 'wheel_scale_' + str(lims_ID) + '.pkl')
    scale_frame = joblib.load(exp_folder + '\Scalers\\' + 'frame_scale_' + str(lims_ID) + '.pkl')
    scale_optical = joblib.load(exp_folder + '\Scalers\\' + 'optical_scale_' + str(lims_ID) + '.pkl')
    scale_angle = joblib.load(exp_folder + '\Scalers\\' + 'angle_scale_' + str(lims_ID) + '.pkl')



    # import wheel data
    dir = os.path.join(exp_folder + '\Wheel\\', 'dxds' + str(lims_ID) + '.pkl' )
    wheel = joblib.load(dir)
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]
    k = first_index[0]
    imp = Imputer(missing_values='NaN', strategy='mean')
    wheel= imp.fit_transform(wheel)
    wheel = scale_wheel.transform(wheel)

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


    behavior_data = ep(exp_folder,lims_ID).get_per_frame_data()

    labels = ["chattering", "trunk_present", "grooming", "trunk_absent", "running",
              "fidget", "tail_relaxed", "tail_tense", "flailing_present", "flailing_absent", "walking"]

    Annotators = ['Mahdi Ramadan', 'Fiona Griffin', 'Nate Berbesque', 'Robert Howard', 'Kyla Mace', 'Twee']


    # length of movie
    limit = int(video_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fps = video_pointer.get(cv2.cv.CV_CAP_PROP_FPS)

    print('creating hf')
    # create hdf file
    hf = h5py.File(exp_folder + '\h5 files\\'+ 'training_data_' + str(lims_ID) + '.h5', 'w')



    while count < limit-first_index[0]:
        # get behavior info for frame

        behavior = ""
        temp = 1
        for item in labels:
            if behavior_data[temp][k] == 1:
                behavior = behavior + " " + item
            temp += 1


        prvs = nex
        frame_data = process_input(prvs)
        frames = scale_frame.transform(frame_data)


        ret, frame = video_pointer.read()
        nex = cv2.cvtColor(frame[160:400, 100:640], cv2.COLOR_BGR2GRAY)

        optical = optical_flow(prvs, nex)
        opticals = scale_optical.transform(optical['mag'])
        angles= scale_angle.transform(optical['ang'])

        vector_data = np.concatenate((np.reshape(wheel[k], (1)), frames, opticals, angles))
        name = 'frame number ' + str(k)
        table = hf.create_dataset(name, data= vector_data, shape=(1, 5713), compression=9)
        table.attrs['behavior'] = behavior

        count += 1
        k += 1

        if count%1000 == 0:
            print (count)

def optical_flow(prvs, next):

    # calculate optical flow and angle of out two frame
    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang= cv2.cartToPolar(flow[..., 0], flow[..., 1])


    # get histograms of optical flow and angles (these are our features)
    mag = process_input(exposure.rescale_intensity(mag, in_range=(-1, 1)))

    #ang = process_input((ang*180/np.pi/2))
    ang = process_input(exposure.rescale_intensity(ang, in_range=(-1, 1)))

    return {'mag': mag, 'ang': ang}

def process_input(input):

    frame_data = []


    for (i, resized) in enumerate(pyramid_gaussian(input, downscale=1.5)):

        fd = hog(resized, orientations=8, pixels_per_cell=(30, 30),
                            cells_per_block=(1, 1), visualise=False)
        frame_data = np.concatenate((frame_data, fd))

        # if the image is too small, break from the loop
        if resized.shape[0] < 100 or resized.shape[1] < 100:
            break


    # for each defined window over the data, bin values and return a
    # properly scaled list of expectation values

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