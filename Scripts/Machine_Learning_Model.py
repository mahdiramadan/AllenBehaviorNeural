
"""machine_learning_train.py by Mahdi Ramadan, 06-18-2016
This program will train an SVM based on the training data inputted, will save model
in a pickle file for later use
"""
import os
import pandas
import sys
# from image_processing import ImageProcessing as ip
from excel_processing import ExcelProcessing as ex
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
from sklearn.feature_selection import RFE
import time
import pdb
from sklearn.decomposition import PCA

def run_svm():


    # hf = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Pre_processed_training_testing_data\\\\1_frames_all_beh_501021421_front.h5'), 'r')
    hftest = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Pre_processed_training_testing_data\\\\1_frames_all_beh_510417261_front.h5'), 'r')
    Lims_ID = '510417261'
    # g1 = hf.get('feature space')
    g2 = hftest.get('feature space')

    test_data = np.array(g2.get('features'))
    # all_data = np.array(g1.get('features'))

    # x_train = all_data[0:50000, 0:3521]
    # y_train = all_data[0:50000, 3521]

    x_test = test_data[:, 0:3521]
    y_test= test_data[:, 3521]

    # estimator = SVC(kernel="rbf")
    # selector = RFE(estimator, step=1)
    # selector = selector.fit(x_train, y_train)
    #
    # x_train_reduced = selector.transform(x_train)
    # x_test_reduced = selector.transform(x_test)



    wheel = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Wheel\dxds'+ str(Lims_ID)+ '.pkl')
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]
    imp = Imputer(missing_values='NaN', strategy='mean')

    # normalize wheel data according to wheel scaler
    wheel = imp.fit_transform(wheel)

    ep = ex("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation", Lims_ID)

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

    neither_vector= []
    for i in  range (len(fidget_vector)):
        if movement_vector[0][i] != 1 and fidget_vector[i] != 1:
            neither_vector.append(1)
        else:
            neither_vector.append(0)

    relevant_fidget = 0
    for i in range(len(fidget_vector)):
        if fidget_vector[i] == 1:
            relevant_fidget += 1

    relevant_movement = 0
    for i in range(len(fidget_vector)):
        if movement_vector[0][i] == 1:
            relevant_movement += 1

    relevant_neither = 0
    for i in range(len(fidget_vector)):
        if neither_vector[i] == 1:
            relevant_neither += 1

    # rows_n = len(x_train)
    # train = int(round(rows_n * 0.6))
    # end = int(round(rows_n * 1.0))
    #
    # X_quick_train = x_train[0:train]
    # X_quick_test = x_train[train:end]
    #
    # y_quick_train = y_train[0:train]
    # y_quick_test = y_train[train:end]
    #
    # # Set the parameters by cross-validation
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-6, 1e-5, 1, 10, 100],
    #                      'C': [0.01, 0.1, 1, 10, 100 ]}]
    #
    # scores = ['precision', 'recall']
    #
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv= 4,
    #                        scoring='%s' % score, n_jobs= 6)
    #     clf.fit(X_quick_train, y_quick_train)
    #
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     for params, mean_score, scores in clf.grid_scores_:
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean_score, scores.std() * 2, params))
    #     print()
    #
    #     print("Detailed classification report:")
    #     print()
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print()
    #     y_true, y_pred = y_quick_test, clf.predict(X_quick_test)
    #     print(classification_report(y_true, y_pred))
    #     print()

    gamma = [ 10e-5]
    C = [ 1 ]


    for g in gamma:
        for c in C:
            # for item in reduction:
            #     if item == 'true':
            #         x_train = x_train_reduced
            #         x_test = x_test_reduced
            print(" training model ")
            # clf = SVC(kernel='rbf', C= c, gamma= g, cache_size= 20000)

            # clf.fit(x_train, y_train)
            print(" saving model ")
            # joblib.dump(clf, 'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\ML Models\one_frames_SVM_all_frames' + str(c)+'_'+str(g)+'.pkl')
            clf = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\ML Models\\one_frames_SVM_all_frames_front1_0.0001.pkl')
            y_pred = []
            print(" predicting data ")

            k = 0

            for index in range (len(x_test)):
                if wheel[index+first_index] > 3:
                    y_pred.append(2)
                else:
                    y_pred.append(clf.predict(x_test[index, :].reshape(1, -1)))

            joblib.dump(y_pred, 'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Predicted behavior\\7261_1_frame_front.pkl')

            true_positive_fidget = 0
            false_positive_fidget = 0
            true_negative_fidget = 0
            false_negative_fidget = 0

            true_positive_movement = 0
            false_positive_movement = 0
            true_negative_movement = 0
            false_negative_movement = 0

            true_positive_neither = 0
            false_positive_neither = 0
            true_negative_neither = 0
            false_negative_neither = 0

            for index in range(len(x_test)):
                if fidget_vector[index+first_index] == 1 and y_pred[index] == 0:
                    true_positive_fidget += 1
                elif fidget_vector[index+first_index] == 1 and y_pred[index] != 0:
                    false_negative_fidget += 1
                elif fidget_vector[index+first_index] != 1 and y_pred[index] != 0:
                    true_negative_fidget += 1
                elif y_pred[index] == 0 and fidget_vector[index+first_index] != 1:
                    false_positive_fidget += 1
            with open('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Results_1_frame_7261.txt', 'w') as file_out:
                # if item == 'true':
                #     print('Feature Selection on')
                #     file_out.write ('Feature Selection on \\\\n')
                print (' for SVM trained with Half data c = ' + str(c) + ' and gamma = ' + str(g)+ ' we get these results:')
                file_out.write (' for SVM trained with c = ' + str(c) + ' and gamma = ' + str(g)+ ' we get these results:\n')
                print ( ' Fidget behavior:\n ')
                file_out.write( ' Fidget behavior:\n')
                print ( ' true positive: ' + str( float(true_positive_fidget)))
                file_out.write(' true positive: ' + str( float(true_positive_fidget) )+ '\n')
                print (' false positive: ' + str(float(false_positive_fidget) ))
                file_out.write (' false positive: ' + str(float(false_positive_fidget) )+ '\n')
                print (' true negative: ' + str(float(true_negative_fidget) ))
                file_out.write (' true negative: ' + str(float(true_negative_fidget) ) + '\n')
                print (' false negative: ' + str(float(false_negative_fidget)))
                file_out.write (' false negative: ' + str(float(false_negative_fidget) )+'\n')

                print()

                for index in range(len(x_test)):
                    if movement_vector[0][index + first_index] == 1 and y_pred[index] == 2:
                        true_positive_movement += 1
                    elif  movement_vector[0][index + first_index] == 1 and y_pred[index] != 2:
                        false_negative_movement += 1
                    elif  movement_vector[0][index + first_index] != 1 and y_pred[index] != 2:
                        true_negative_movement += 1
                    elif y_pred[index] == 2 and  movement_vector[0][index + first_index] != 1:
                        false_positive_movement += 1
                # if item == 'true':
                #     print('Feature Selection on')
                #     file_out.write ('Feature Selection on \\\\n')
                print (' for SVM trained with c = ' + str(c) + ' and gamma = ' + str(g) + ' we get these results:')
                file_out.write(' for SVM trained with c = ' + str(c) + ' and gamma = ' + str(g) + ' we get these results:\n')
                print (' Movement behavior: \n')
                file_out.write( 'Movement behavior')
                print (' true positive: ' + str(float(true_positive_movement) ))
                file_out.write (' true positive: ' + str(float(true_positive_movement) )+ '\n')
                print (' false positive: ' + str(float(false_positive_movement)))
                file_out.write (' false positive: ' + str(float(false_positive_movement)) + '\n')
                print (' true negative: ' + str(float(true_negative_movement)))
                file_out.write (' true negative: ' + str(float(true_negative_movement) )+ '\n')
                print (' false negative: ' + str(float(false_negative_movement) ))
                file_out.write (' false negative: ' + str(float(false_negative_movement) )+ '\n')

                print()

                for index in range(len(x_test)):
                    if neither_vector[index + first_index] == 1 and y_pred[index] == 1:
                        true_positive_neither += 1
                    elif neither_vector[index + first_index] == 1 and y_pred[index] != 1:
                        false_negative_neither += 1
                    elif neither_vector[index + first_index] != 1 and y_pred[index] != 1:
                        true_negative_neither += 1
                    elif y_pred[index] == 1 and neither_vector[index + first_index] != 1:
                        false_positive_neither += 1

                # if item == 'true':
                #     print('Feature Selection on')
                #     file_out.write ('Feature Selection on \\\\n')
                print (' for SVM trained with c = ' + str(c) + ' and gamma = ' + str(g) + ' we get these results:')
                file_out.write (' for SVM trained with c = ' + str(c) + ' and gamma = ' + str(g) + ' we get these results:\n')
                print (' Neither behavior: ')
                file_out.write( ' Neither behavior\n')
                print (' true positive: ' + str(float(true_positive_neither) ))
                file_out.write (' true positive: ' + str(float(true_positive_neither))+'\n')
                print (' false positive: ' + str(float(false_positive_neither)))
                file_out.write (' false positive: ' + str(float(false_positive_neither) )+'\n')
                print (' true negative: ' + str(float(true_negative_neither) ))
                file_out.write (' true negative: ' + str(float(true_negative_neither) )+'\n')
                print (' false negative: ' + str(float(false_negative_neither) ))
                file_out.write (' false negative: ' + str(float(false_negative_neither) )+'\n')
                file_out.write ('\n')
                file_out.write('\n')
                print()

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
	
    # # set limsID of video data to train on
    # # '503412730', '497060401','502741583', '501004031', '500860585', '501560436'
    run_svm()
