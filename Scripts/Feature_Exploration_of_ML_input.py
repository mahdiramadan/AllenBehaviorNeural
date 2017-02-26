
"""machine_learning_train.py by Mahdi Ramadan, 06-18-2016
This program will train an SVM based on the training data inputted, will save model
in a pickle file for later use
"""
import os
import pandas
import sys
# from image_processing import ImageProcessing as ip
from excel_processing import ExcelProcessing as ep
import pickle
from sklearn.svm import NuSVC
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from multiprocessing import Process
from sklearn.cross_validation import train_test_split
from math import isnan
from sklearn.externals import joblib
import scipy
import collections
from sklearn import datasets, linear_model
from sklearn.preprocessing import Imputer
import time
import pdb
from sklearn.decomposition import PCA

def run_svm( data):

    plt.hist(data, bins='auto', range = (-10,15))  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    # final_data = np.float32(final_data)
    # scipy.io.savemat('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Matlab\\\\60_frames_2000_fidget_neither_503412730_497060401_502741583_501004031_500860585_501560436_503412730_501773889.mat', {'features':final_data})
    # rows_n = len(final_data)
    # train = int(round(rows_n*0.75))
    #
    # a = set(y_vector)
    # print(a)
    zero = 0
    one = 0
    two = 0





def get_data(lims_ID, ep):

    # # import h5py file for each lims ID video
    # hf = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\h5 files/training_data_pervectornorm' + str(lims_ID) + '.h5'), 'r')
    # # get a list of all keys in the file dictionary, where in keys correspond to frame number
    # data_tables = hf.keys()
    # create variable k, which will keep track of current frame number
    k = 0

    # important wheel data, get rid of NaN values, and find first non-NaN index. This index will be our starting point
    # (don't want to train on frames without a wheel value, since it is part of our feature vector)
    wheel = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Wheel\dxds'+ str(lims_ID)+'.pkl')
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]
    imp = Imputer(missing_values='NaN', strategy='mean')

    # normalize wheel data according to wheel scaler
    wheel = imp.fit_transform(wheel)


    label = 'fidget'
    index = ep.get_labels().index(label) + 1
    fidget_vector = np.array(ep.get_per_frame_data()[index])

    label = 'walking'
    index = ep.get_labels().index(label) + 1
    walking = np.array(ep.get_per_frame_data()[index])

    label = 'running'
    index = ep.get_labels().index(label) + 1
    running = np.array(ep.get_per_frame_data()[index])

    movement_vector = []
    movement_vector.append([sum(x) for x in zip(walking, running)])

    # set your desired starting frame relative to first_index
    start_Frame = 0

    # store length of feature vectors
    # number_of_features = 5713
    # # set desired frames per training block
    # frames_per_block = 60


    # update k to fist_index plus any desired start frame offset!!!
    k = start_Frame + first_index

    values = []
    for i in range(k, len(movement_vector[0])):
        if movement_vector[0][i] == 1:
            try:
                values.append(float(wheel[i]))
            except:
                continue



    return { 'values': values}

def beh_present(string):
    # method will return the appropriate behavior label of interest based on its presence
    # behaviors must be mutually exclusive for method to work
    if "fidget" in string:
        return "fidget "
    elif "running" in string or "walking" in string:
        return "movement "
    else:
        return "neither "

def mode_beh(string):
    # method will return label for most frequent behavior of interest for a block of frames
    # count how many fidget and movement (walking and running) behaviors are present
    fidget = string.count("fidget")
    movement = string.count("movement")
    neither = string.count("neither")

    # based on most frequent behavior. return label (0 is fidget, 1 is movement, 2 is neither)
    if fidget > movement and fidget > neither:
        return 0
    elif movement > fidget and movement > neither:
        return 2
    else:
        return 1


def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # set limsID of video data to train on
    # '503412730', '497060401','502741583', '501004031', '500860585', '501560436'
    lims_ID = ['503412730', '497060401','502741583', '501004031', '500860585', '501560436']
    # initialize training data and data label arrays, as well a a loop counter
    values = []
    t= 0

    # for each limd_ID, get trainign and label data, and combine it with previous lims ID training and label data
    for itm in lims_ID:
        ex = ep("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation", itm)
        data = get_data(itm, ex)
        if t == 0:
            values = data['values']
            print(itm + ' video done')

        else:
            temp = data['values']
            values = np.concatenate((values, temp))
            print(itm + ' video done')

        t += 1

    print('feature processing finished')
    run_svm(values)