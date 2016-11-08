
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
from sklearn import datasets, linear_model
from sklearn.preprocessing import Imputer
import time


def run_svm(final_data, y_vector):
    rows_n = len(final_data)
    train = int(round(rows_n*0.8))


    X_train = final_data[0:train]
    X_test = input['feature_data'][train:rows_n]


    y_train = y_vector[0:train]
    y_test = y_vector[train:rows_n]


    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 5e-3, 1e-2],
                         'C': [1,5, 10]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
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
    hf = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\h5 files/training_data_' + str(lims_ID) + '.h5'), 'r')
    data_tables = hf.keys()
    k = 0

    y_train = []
    y_fid = []
    feature_data= []

    wheel = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Wheel\dxds'+ str(lims_ID)+'.pkl')
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]

    start_Frame = 70000
    end_frame = 80000

    # feature vectors start at first non NaN wheel index!!!
    k = start_Frame + first_index

    data_length = len(data_tables)
    count = 0
    beh_type = ""

    while data_length/7 >= 1:

        if count == 0:
            feature_data = np.array(hf.get('frame number ' + str(int(k))))
            beh_type += hf.get('frame number ' + str(int(k))).attrs['behavior']
            for index in range(k+1, k+7):
                feature_data = np.hstack((feature_data, np.array(hf.get('frame number ' + str(int(index))))))
                beh_type += hf.get('frame number ' + str(int(k))).attrs['behavior']
            count += 1
            k += 7
            data_length = data_length - 7
            y_train.append(mode_beh(beh_type))
            beh_type = ""

        else:
            temp = np.array(hf.get('frame number ' + str(int(k))))
            beh_type += hf.get('frame number ' + str(int(k))).attrs['behavior']
            for index in range(k+1, k + 7):
                temp = np.hstack((temp ,np.array(hf.get('frame number ' + str(int(index))))))
                beh_type += hf.get('frame number ' + str(int(k))).attrs['behavior']

            feature_data = np.vstack((feature_data,temp))
            k += 7
            data_length = data_length - 7
            y_train.append(mode_beh(beh_type))
            beh_type = ""

    return {'feature_data': feature_data, 'y_train': y_train}

def mode_beh(string):
    l = len(string.split())
    fidget = string.count("fidget")
    movement = string.count("walking") + string.count("running")
    neither = len(string.split()) - (fidget+movement)

    if fidget > movement or fidget > neither:
        return 0
    elif movement > fidget or movement > neither:
        return 1
    else:
        return 2



def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    lims_ID = ['501560436']
    final_data = []
    y_train = []
    feature_data=[]
    y_fid=[]
    wheel=[]
    t= 0

    for itm in lims_ID:
        data = get_data(itm)
        if t == 0:
            y_train= data['y_train']
            feature_data = data['fidget_data']
        else:
            vector = data['final_data']
            feature_data = np.vstack((feature_data, vector))
            y_train = np.concatenate((y_train, data['y_train']))
        t += 1

    print('feature processing finished')
    p = Process(target = run_svm(final_data, y_train), args = (final_data, y_train))