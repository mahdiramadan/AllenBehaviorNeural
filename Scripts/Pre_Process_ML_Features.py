
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

def run_svm(final_data, y_vector, y_track):
    hf = h5py.File('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Pre_processed_training_testing_data\\\\300_frames_all_beh_all_front_blocks.h5', 'w')
    g = hf.create_group('feature space')
    a= len(final_data)
    try:
        table = g.create_dataset('features', data=final_data, shape=(len(final_data), 3522))
    except:
        table = g.create_dataset('features', data=final_data, shape=(len(y_vector), 3522))
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

    # import h5py file for each lims ID video
    hf = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\h5 files/training_data_front_of_mouse' + str(lims_ID) + '.h5'), 'r')
    # get a list of all keys in the file dictionary, where in keys correspond to frame number
    data_tables = hf.keys()
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
    number_of_features = 3521
    # set desired frames per training block
    frames_per_block = 60


    # update k to fist_index plus any desired start frame offset!!!
    k = start_Frame + first_index



    relevant_fidget = 0
    relevant_movement = 0
    relevant_neither = 0

    max_fidget = 0
    max_neither = 0
    max_movement = 0

    for i in range (k, len(data_tables), frames_per_block):
        for k in range (k, k + frames_per_block):
            if fidget_vector[k] == 1:
                relevant_fidget += 1
            if movement_vector[0][k] == 1:
                relevant_movement += 1
            if movement_vector[0][k] == 0 or fidget_vector[k] == 0:
                relevant_neither += 1

        if relevant_fidget >= relevant_movement and relevant_fidget >= relevant_neither:
            max_fidget += 1
        elif relevant_movement > relevant_fidget and relevant_movement > relevant_neither:
            max_movement += 1
        else:
            max_neither += 1

        relevant_fidget = 0
        relevant_movement = 0
        relevant_neither = 0




    # calculate how many frames the training data consists of
    if len(data_tables) - start_Frame > 0:
        data_length = len(data_tables) - start_Frame
    else:
        data_length = 0

    #initialize a counter
    count = 0
    flex_count = 0
    rel_count = 0
    # initialize behavior string
    beh_type = ""

    # initialize feature data and label data arrays
    y_train = []
    y_track = []
    feature_data = np.empty((data_length, number_of_features+1), dtype = float)
    print(feature_data.dtype)
    neither_count = 0
    fidget_count = 0
    movement_count= 0


    # while we have block_zframes left for each training block, add feature vector to training data and label to training label
    while k+frames_per_block < len(data_tables):
        # get data of first frame
        # for each frame, add the associated behavior attributes
        # beh = mode_beh(beh_present(hf.get('frame number ' + str(int(k+15))).attrs['behavior']))

        fidget_present = 0

        for i in range(k, k+frames_per_block):
            if fidget_vector[i] == 1:
                fidget_present += 1

        movement_present = 0

        for i in range(k, k+frames_per_block):
            if movement_vector[0][i] == 1:
                movement_present += 1

        neither_present = 0

        for i in range(k, k+frames_per_block):
            if movement_vector[0][i] == 0 or fidget_vector[i] == 0:
                neither_present += 1



        if fidget_present >= movement_present and fidget_present >= neither_present:
            beh = 0
        elif movement_present > fidget_present and movement_present > neither_present:
            beh = 2
        else:
            beh = 1
        # Concatenate the feature vectors of the next block size frames
        temp = np.mean([np.array(hf.get('frame number ' + str(int(j)))) for j in range (k, k+ frames_per_block)], axis=0)


        # set the corrrespoding row of feature data to concatenated frame features of size block_size
        if beh == 1 and neither_count <= max_fidget:
            feature_data[count, 0:number_of_features] = temp
            feature_data[count, number_of_features] = beh
            y_train.append(beh)
            count += 1
            neither_count += 1
        elif beh == 0:
            feature_data[count, 0:number_of_features] = temp
            feature_data[count, number_of_features] = beh
            y_train.append(beh)
            count += 1
            fidget_count += 1
        elif beh == 2:
            count += 1
            movement_count += 1
        # iterate current frame
        k += frames_per_block
        # update how many frames are left to process
        data_length = data_length - 1
        # based on most frequent behavior in behavior attribute string, add label to training label for training block
        # reset behavior attribute for next block
        beh_type = ""

        if count%1000 == 0:
            print(count)




    return {'feature_data': feature_data, 'y_train': y_train, 'y_track': y_track, 'number': fidget_count+ neither_count + movement_count}

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
    lims_ID = [ '502741583', '501004031', '500860585', '501560436']
    # initialize training data and data label arrays, as well a a loop counter
    y_train = []
    y_track = []
    feature_data=[]
    t= 0

    # for each limd_ID, get trainign and label data, and combine it with previous lims ID training and label data
    for itm in lims_ID:
        ex = ep("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation", itm)
        data = get_data(itm, ex)
        if t == 0:
            y_train = data['y_train']
            y_track = data['y_track']
            feature_data = data['feature_data'][0:len(y_train)]
            print(itm + ' video done')
            print ( data['number'])
        else:
            y_vector = data['y_train']
            y_train = np.concatenate((y_train, y_vector))
            y_track = np.concatenate((y_track, data['y_track']))
            vector = data['feature_data'][0:len(y_vector)]
            feature_data = np.vstack((feature_data, vector))
            print(itm + ' video done')

        t += 1

    print('feature processing finished')
    run_svm(feature_data, y_train, y_track)