
"""machine_learning.py by Mahdi Ramadan, 06-18-2016
This program will be used for machine learning fitting
and prediction
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
from sklearn.preprocessing import Imputer
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
import HDDA
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import PCA
from sklearn import mixture

def run_svm(final_data, y_vector):

    # rows_n = len(input['feature_data'])
    # train = int(round(rows_n*0.8))
    # test = int(rows_n - train)
    #
    # X_train = input['feature_data'][0:train]
    # X_test = input['feature_data'][train:rows_n]
    #
    #
    # y_train = input['labels'][0:train]
    # y_test = input['labels'][train:rows_n]
    # X_train, X_test, y_train, y_test = train_test_split(final_data, y_train, test_size=0.30, random_state= 28)

    X_train = final_data[0:100000]
    y_train = y_vector[0:100000]

    X_test = final_data[100000:150000]
    y_test = y_vector[100000:150000]


    # #Parameters for HDDA
    # MODEL = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
    # C = [3]  # For the example with do not fit the number of classes
    # th = [0.01, 0.05, 0.1, 0.2]  # The threshold for the Cattel test

    model = HDDA.HDGMM()

    model.fit(X_train, y_train, param = {'C': 2, 'th':0.01, 'MODEL': 'M4'})
    yp = model.predict(X_test)

    print(classification_report(y_test, yp))

    # joblib.dump(yp, 'HDDA_fidget_vs_movement_vs_neither.pkl')


def get_data(ep, lims_ID):
    hf = h5py.File(('data_' + str(lims_ID) + '.h5'), 'r')
    k = 0

    y_train = []
    dimension = 260*540
    wheel = joblib.load('C:\Users\mahdir\Desktop\Mahdi files\Wheel\dxds'+ str(lims_ID)+'.pkl')
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]

    start_Frame = 15000
    end_frame = 65000

    # feature vectors start at first non NaN wheel index!!!
    k = start_Frame + first_index

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

    group = hf.get('feature space')
    data = wheel[start_Frame+first_index : end_frame+first_index]


    for item in range(start_Frame, end_frame):
        if movement_vector[0][k] == 1:
            y_train.append(0)
            # elif movement_vector[0][k] == 1:
            #     y_train.append(1)
        else:
            y_train.append(1)

        k += 1
    return {'final_data': data, 'y_train': y_train}

def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    lims_ID = ['501560436', '501021421','500860585' ]
    final_data = []
    y_train = []
    t= 0

    for itm in lims_ID:
        exl = ep("C:\Users\mahdir\Desktop\Mahdi files", itm)
        data = get_data(exl, itm)
        if t == 0:
            final_data = data['final_data']
            y_train= data['y_train']
        else:
            vector = data['final_data']
            final_data = np.vstack((final_data, vector))
            y_train = np.concatenate((y_train, data['y_train']))
        t += 1

    print('feature processing finished')
    p = Process(target = run_svm(final_data, y_train), args = (final_data, y_train))