
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
from sklearn.lda import LDA

def run_svm(final_data, y_vector, feature_data, y_fid, wheel):

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

    X_train = final_data[0:30000]
    y_train = y_vector[0:30000]

    X_test = final_data[30000:40000]
    y_test = y_vector[30000:40000]


    # # # # clf = RandomForestClassifier(verbose=3)
    # n_estimators = 10
    # clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma = 0.001, C = 10),n_estimators = n_estimators), n_jobs = -1)

    # clf = LDA()
    # clf.fit(X_train, y_train)
    # joblib.dump(clf, 'clf_movement_nomovement_optical_wheel_SVM.pkl', compress=1)

    # clf = SVC(C=10, gamma=0.001, kernel='rbf', class_weight= 'auto' )
    # clf.fit(feature_data[0:30000], y_fid[0:30000])
    # joblib.dump(clf, 'clf_fidget_vs_all.pkl', compress=1)
    y_final= []
    wheel = wheel[0:10000]

    clf = joblib.load('clf_fidget_vs_all.pkl')

    for i in range(len(X_test)):
        if wheel[i] > 6:
            y_final.append(1)

        else:
            y_final.append(int(clf.predict(X_test[i])))


    print(classification_report(y_test, y_final))

    joblib.dump(y_final, 'y_pred_fid_move_nomove_0585_wheel_split_2nd.pkl', compress =1)

    # # clf = joblib.load('filename.pk1')

def get_data(ep, lims_ID):
    hf = h5py.File(('data_' + str(lims_ID) + '.h5'), 'r')
    k = 0

    y_train = []
    y_fid = []
    feature_data= []

    dimension = 260*540
    wheel = joblib.load('C:\Users\mahdir\Desktop\Mahdi files\Wheel\dxds'+ str(lims_ID)+'.pkl')
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]

    start_Frame = 70000
    end_frame = 80000

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
    # data = np.concatenate((np.array(group.get('features'))[start_Frame:end_frame, 0:1], np.array(group.get('features'))[start_Frame:end_frame, 1441:2881]), axis =1)
    data_W = wheel[start_Frame + first_index:end_frame + first_index]
    data = np.array(group.get('features'))[start_Frame:end_frame]
    y_move =[]
    t = 0
    t1 = time.time()
    for i in range(len(data_W)):

        if data_W[i] > 6:
            y_move.append(1)

        else:
            y_move.append(0)

            if fidget_vector[k]== 1:
                y_fid.append(0)
                if t ==0:
                    feature_data = data[i]
                else:
                    feature_data = np.vstack((feature_data, data[i]))
            else:
                y_fid.append(2)
                if t == 0:
                    feature_data = data[i]
                else:
                    feature_data = np.vstack((feature_data, data[i]))
        t+=1
        k+=1

    k = start_Frame + first_index
    for item in range(start_Frame, end_frame):
        if fidget_vector[k] == 1:
            y_train.append(0)
        elif movement_vector[0][k] == 1:
            y_train.append(1)

        else:
            y_train.append(2)

        k += 1
    t2 = time.time()
    print(t2-t1)
    return {'final_data': data, 'y_train': y_train, 'fidget_data':feature_data, 'y_fid':y_fid, 'wheel':data_W}

def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    lims_ID = ['501560436', '502741583','509904120', '500860585']
    final_data = []
    y_train = []
    feature_data=[]
    y_fid=[]
    wheel=[]
    t= 0

    for itm in lims_ID:
        exl = ep("C:\Users\mahdir\Desktop\Mahdi files", itm)
        data = get_data(exl, itm)
        if t == 0:
            final_data = data['final_data']
            y_train= data['y_train']
            feature_data = data['fidget_data']
            y_fid = data['y_fid']
            wheel = data['wheel']
        else:
            vector = data['final_data']
            final_data = np.vstack((final_data, vector))
            y_train = np.concatenate((y_train, data['y_train']))
            feature_data = np.concatenate((feature_data, data['fidget_data']))
            y_fid = np.concatenate((y_fid, data['y_fid'] ))
            wheel = np.concatenate((wheel, data['wheel']))
        t += 1

    print('feature processing finished')
    p = Process(target = run_svm(final_data, y_train, feature_data, y_fid, wheel), args = (final_data, y_train, feature_data, y_fid, wheel))