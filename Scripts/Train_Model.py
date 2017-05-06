import os
import pandas
import sys
from lims_database import LimsDatabase as ld
# from image_processing import ImageProcessing as ip
from excel_processing import ExcelProcessing as ex
import pickle
from sklearn.svm import NuSVC
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pylab import *
from math import log
from scipy import stats
from skimage.feature import hog
from skimage import data, color, exposure
from scipy.optimize import curve_fit
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from multiprocessing import Process
import matplotlib.mlab as mlab
from sklearn.cross_validation import train_test_split
from math import isnan
from sklearn.externals import joblib
import scipy
import collections
from sklearn import datasets, linear_model
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import RFE
import time
import pdb
from psycopg2 import connect
from lims_database import LimsDatabase as ld
from Sync_Camera_Stimulus import Get_Wheel as gw

def train_svm (data):


    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv= 4,
                           scoring='%s' % score, n_jobs= 6)
        clf.fit(data[0:2], data[2])

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
        y_true, y_pred = data[2], clf.predict(data[0:2])
        print(classification_report(y_true, y_pred))
        print()

def run (ids , ann_data, t, v_limits):
    # create CAM stimulus definition visual
    count = 0


    video_directory = ld(ids).get_video_directory()
    video_file = False

    for file in os.listdir(video_directory):
        # looks for the pkl file and makes the directory to it
        if file.endswith("-0.avi") and file.startswith(ids):
            video_file = os.path.join(video_directory, file)
    if video_file == False:
        print('Movie file not found')
        print(str(ids))

    video_pointer = cv2.VideoCapture(video_file)
    # set video pointer to first frame, and read first frame

    x1 = 180
    x2 = 360
    y1 = int(v_limits[(t*2)])
    y2 = int(v_limits[((t*2)+1)])

    if os.path.isfile('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Wheel\\wheel' + str(ids) + '.pkl'):
        wheel = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Wheel\\wheel' + str(ids) + '.pkl')
    else:
        print ('wheel data not found')
        print(str(ids))

    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]
    imp = Imputer(missing_values='NaN', strategy='mean')
    # normalize wheel data according to wheel scaler
    wheel = imp.fit_transform(wheel)
    k = first_index[0]



    # set video pointer to first frame, and read first frame
    video_pointer.set(1, k)
    ret, frame = video_pointer.read()


    # crops and converts frame into desired format
    frame = cv2.cvtColor(frame[x1:x2, y1:y2], cv2.COLOR_BGR2GRAY)

    # initialize prvs and nex that will keep track of the previous and current frame(frame difference needed to
    # calculate optical flow)
    prvs = frame
    nex = frame

    # initialize loop counter, vectors to keep track of data
    count = 0
    mod = 0
    opticals = []

    frame_limit = len(wheel)

    feature_data = np.empty((len(wheel), 3), dtype=float)

    while k < frame_limit and ret:
        prvs = nex
        ret, frame = video_pointer.read()
        nex = cv2.cvtColor(frame[x1:x2, y1:y2], cv2.COLOR_BGR2GRAY)
        optical = abs(optical_flow(prvs, nex)['mag'])
        feature_data[count, 0] = np.mean(optical)
        feature_data[count, 1] = wheel[k]
        feature_data[count, 2] = int(ann_data[count])

        k += 1
        count += 1

    return {'feature_data': feature_data, 'count': count}

def optical_flow(prvs, next):


    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 30, 3, 5, 1.2, 0)
    mag, ang= cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # get histograms of optical flow and angles (these are our features)
    mag = process_input(exposure.rescale_intensity((1/mag), in_range=(-1, 1)))

    #ang = process_input((ang*180/np.pi/2))


    return {'mag': mag, 'ang': ang}

def process_input(input):

    # show_frame(resized)
    fd, h  = hog(input, orientations= 8, pixels_per_cell=(15, 15),
                        cells_per_block=(1, 1), visualise= True)

    return h

def get_all_data (lims_ids):
    data = [[] for _ in range(len(lims_ids))]
    k = 0
    for id in lims_ids:
        id = id.strip()
        try:
            data[k] = joblib.load ('\\\\aibsdata\ophysdev\oPhysQC\\beh_annotation\\Annotation Files\\30_frame_threshold_blocked_' + str(id) + '.pkl')
            k += 1
        except:
            print(str(id))

    return data

if __name__ == '__main__':

    text_file = open("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\temp.txt", "r")
    lims_ids = text_file.read().split(',')
    # print (lims_ids[5] + lims_ids[12] + lims_ids[14] + lims_ids[25] +lims_ids[34] + lims_ids[41] +lims_ids[46] + lims_ids[49]  +lims_ids[54] + lims_ids[55]   )
    ann_data = get_all_data(lims_ids)
    #
    y_train = []
    y_track = []
    feature_data = []
    t = 0
    v_limits = open("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\video_limits.txt", "r")
    v_limits = v_limits.read().split(',')

    # for each limd_ID, get trainign and label data, and combine it with previous lims ID training and label data
    for itm in lims_ids:
        itm = itm.strip()
        data = run(itm, ann_data[t], t, v_limits)
        if t == 0:
            y_train = data['count']
            feature_data = data['feature_data'][0:y_train]
            print(itm + ' video done')

        else:
            y_train = data['count']
            vector = data['feature_data'][0:y_train]
            feature_data = np.vstack((feature_data, vector))
            print(itm + ' video done')

        t += 1

    train_svm (feature_data)
