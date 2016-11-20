
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
import collections
from sklearn import datasets, linear_model
from sklearn.preprocessing import Imputer
import time
import pdb

def run_svm(final_data, y_vector, y_track):

    rows_n = len(final_data)
    train = int(round(rows_n*0.6))

    a = set(y_vector)
    print(a)
    zero = 0
    one = 0
    two = 0

    for i in range(0, len(y_vector)):
        if (y_vector[i] == 0):
            zero += 1
        elif (y_vector[i] == 1):
            one += 1
        else:
            two += 1
    print(zero , one , two)



    X_train = final_data[0:train]
    X_test = final_data[train:rows_n]


    y_train = y_vector[0:train]
    y_test = y_vector[train:rows_n]


    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-6, 1e-5, 1e-4],
                         'C': [0.001, 0.01,  0.1 , 10]}]

    scores = ['f1_weighted']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=2,
                           scoring='%s' % score, n_jobs=-1)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

        # # # clf = RandomForestClassifier(verbose=3)
        # # clf = SVC(kernel='linear', C = 0.1, verbose = 2)
        #
        # clf = neighbors.KNeighborsClassifier(2)
        # clf.fit(X_train, y_train)
        #
        # joblib.dump(clf, 'clf.pkl')
        #
        # y_true, y_pred = y_test, clf.predict(X_test)
        # print(classification_report(y_true, y_pred))




def get_data(lims_ID):

    # import h5py file for each lims ID video
    hf = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\h5 files/training_data_pervectornorm' + str(lims_ID) + '.h5'), 'r')
    # get a list of all keys in the file dictionary, where in keys correspond to frame number
    data_tables = hf.keys()
    # create variable k, which will keep track of current frame number
    k = 0

    # important wheel data, get rid of NaN values, and find first non-NaN index. This index will be our starting point
    # (don't want to train on frames without a wheel value, since it is part of our feature vector)
    wheel = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Wheel\dxds'+ str(lims_ID)+'.pkl')
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]

    # set your desired starting frame relative to first_index
    start_Frame = 0

    # store length of feature vectors
    number_of_features = 5713
    # set desired frames per tarining block
    frames_per_block = 1


    # update k to fist_index plus any desired start frame offset!!!
    k = start_Frame + first_index

    # calculate how many frames the training data consists of
    data_length = len(data_tables) - start_Frame
    #initialize a counter
    count = 0
    # initialize behavior string
    beh_type = ""

    # initialize feature data and label data arrays
    y_train = []
    y_track = []
    feature_data = np.empty((data_length/frames_per_block, number_of_features*frames_per_block), dtype = float)
    neither_count = 0

    # while we have block_zframes left for each training block, add feature vector to training data and label to training label
    while data_length/frames_per_block >= 1:
        # get data of first frame
        temp = np.array(hf.get('frame number ' + str(int(k))))
        # for each frame, add the associated behavior attributes
        beh_type += beh_present(hf.get('frame number ' + str(int(k))).attrs['behavior'])

        # Concatenate the feature vectors of the next block size frames
        for index in range(k+1, k+frames_per_block):
            temp = np.hstack((temp, np.array(hf.get('frame number ' + str(int(index))))))
            # for each frame, add the associated behavior attributes
            beh_type += beh_present(hf.get('frame number ' + str(int(k))).attrs['behavior'])
        # set the corrrespoding row of feature data to concatenated frame features of size block_size
        if mode_beh(beh_type) != 1 and neither_count <= 2000:
            feature_data[count] = temp
            y_train.append(mode_beh(beh_type))
            count += 1
        elif mode_beh(beh_type) != 1 and neither_count > 2000:
            if mode_beh(beh_type) == 2:
                neither_count += 1
            else:
                feature_data[count] = temp
                y_train.append(mode_beh(beh_type))
                # iterate loop counter
                count += 1
        # iterate current frame
        k += frames_per_block
        # update how many frames are left to process
        data_length = data_length - frames_per_block
        # based on most frequent behavior in behavior attribute string, add label to training label for training block
        if mode_beh(beh_type) == 2:
            neither_count += 1
        y_track.append(mode_beh(beh_type))
        # reset behavior attribute for next block
        beh_type = ""


    return {'feature_data': feature_data, 'y_train': y_train, 'y_track': y_track}

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
        return 1
    else:
        return 2



def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # set limsID of video data to train on
    lims_ID = ['501560436', '500860585', '497060401']
    # initialize training data and data label arrays, as well a a loop counter
    y_train = []
    y_track = []
    feature_data=[]
    t= 0

    # for each limd_ID, get trainign and label data, and combine it with previous lims ID training and label data
    for itm in lims_ID:
        data = get_data(itm)
        if t == 0:
            y_train = data['y_train']
            y_track = data['y_track']
            feature_data = data['feature_data'][0:len(y_train)]
            print(itm + ' video done')
        else:
            y_vector = data['y_train']
            y_train = np.concatenate((y_train, y_vector))
            y_track = np.concatenate((y_track, data['y_track']))
            vector = data['feature_data'][0:len(y_vector)]
            feature_data = np.vstack((feature_data, vector))
            print(itm + ' video done')
        t += 1

    print('feature processing finished')

    # Now that we have all of our training data, feed data to method that will traing ML model
    p = Process(target = run_svm(feature_data, y_train, y_track), args = (feature_data, y_train, y_track))