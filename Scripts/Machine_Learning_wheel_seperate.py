
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

def run_svm(lims_id):

    hf = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Pre_processed_training_testing_data\\\\60_frames_fidget_neither_maxed_front.h5'), 'r')
    hftest = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Pre_processed_training_testing_data\\\\60_frames_all_beh_501021421_front.h5'), 'r')

    g1 = hf.get('feature space')
    g2 = hftest.get('feature space')

    test_data = np.array(g2.get('features'))
    all_data = np.array(g1.get('features'))

    x_train = all_data[:, 0:3521]
    y_train = all_data[:, 3521]

    x_test = test_data[:, 0:3521]
    y_test= test_data[:, 3521]

    # estimator = SVC(kernel="rbf")
    # selector = RFE(estimator, step=1)
    # selector = selector.fit(x_train, y_train)
    #
    # x_train_reduced = selector.transform(x_train)
    # x_test_reduced = selector.transform(x_test)



    wheel = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Wheel\dxds501021421.pkl')
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]
    imp = Imputer(missing_values='NaN', strategy='mean')

    # normalize wheel data according to wheel scaler
    wheel = imp.fit_transform(wheel)

    ep = ex("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation", '501021421')

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
    # train = int(round(rows_n * 0.15))
    # end = int(round(rows_n * 0.2))
    #
    # X_quick_train = x_train[0:train]
    # X_quick_test = x_train[train:end]
    #
    # y_quick_train = y_train[0:train]
    # y_quick_test = y_train[train:end]
    #
    # # Set the parameters by cross-validation
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-6, 1e-5, 1e-4],
    #                      'C': [0.01, 0.1 ]}]
    #
    # scores = ['precision', 'recall']
    #
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3,
    #                        scoring='%s' % score, n_jobs=-1)
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

    gamma = [ 10e-5 ]
    C = [0.1, 1, 10]
    print ('60 frames')

    for g in gamma:
        for c in C:
            # for item in reduction:
            #     if item == 'true':
            #         x_train = x_train_reduced
            #         x_test = x_test_reduced
            print(" training model ")
            clf = SVC(kernel='rbf', C= c, gamma= g, cache_size= 200000)

            clf.fit(x_train, y_train)
            print(" saving model ")
            joblib.dump(clf, 'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Pre_processed_training_testing_data\\thirty_frames_SVM_all_frames_front' + str(c)+'_'+str(g)+'.pkl')
            # clf = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\ML Models\clf_movement_fidget_only_0585no_wheel_larger_data.pkl')
            y_pred = []
            print(" predicting data ")

            k = 0

            for index in range (len(x_test)):
                if wheel[index+first_index] > 3:
                    y_pred.append(2)
                else:
                    y_pred.append(clf.predict(x_test[index, :].reshape(1, -1)))

            joblib.dump(y_pred, 'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Predicted behavior\\thirty_frame_front_1421' + str(c) + str(gamma) + '.pkl')

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
            with open('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Results_30_frame_1421.txt', 'w') as file_out:
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






    # hf = h5py.File('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Matlab\\\\60_frames_maxed_fidget_neither_all.h5', 'w')
    # g = hf.create_group('feature space')
    # a= len(final_data)
    # try:
    #     table = g.create_dataset('features', data=final_data, shape=(len(final_data), 5714))
    # except:
    #     table = g.create_dataset('features', data=final_data, shape=(len(y_vector), 5714))
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
    #

    # X_train, X_test, y_train, y_test = train_test_split(final_data, y_vector, test_size=0.35, random_state=32)
    #
    # # pca = PCA(n_components= 10)
    # # pca.fit(final_data)
    # #
    # # X_train = pca.transform(X_train)
    # # X_test = pca.transform(X_test)
    #
    #
    #
    # # # Set the parameters by cross-validation
    # # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [ 1e-5, 1e-4],
    # #                      'C': [ 0.1 , 10, 100]}]
    # #
    # # scores = ['f1_weighted']
    # #
    # # for score in scores:
    # #     print("# Tuning hyper-parameters for %s" % score)
    # #     print()
    # #
    # #     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=2,
    # #                        scoring='%s' % score, n_jobs=-1)
    # #     clf.fit(X_train, y_train)
    # #
    # #     print("Best parameters set found on development set:")
    # #     print()
    # #     print(clf.best_params_)
    # #     print()
    # #     print("Grid scores on development set:")
    # #     print()
    # #     for params, mean_score, scores in clf.grid_scores_:
    # #         print("%0.3f (+/-%0.03f) for %r"
    # #               % (mean_score, scores.std() * 2, params))
    # #     print()
    # #
    # #     print("Detailed classification report:")
    # #     print()
    # #     print("The model is trained on the full development set.")
    # #     print("The scores are computed on the full evaluation set.")
    # #     print()
    # #     y_true, y_pred = y_test, clf.predict(X_test)
    # #     print(classification_report(y_true, y_pred))
    # #     print(
    #
    #     # # # clf = RandomForestClassifier(verbose=3)
    # clf = SVC(kernel='rbf', C = 10, gamma = 1e-5 )
    #     #
    #     # clf = neighbors.KNeighborsClassifier(2)
    # clf.fit(X_train, y_train)
    #     #
    # joblib.dump(clf, 'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\ML Models\sixty_f_block_fidget_neither_test_many.pkl')
    # #
    # y_true, y_pred = y_test, clf.predict(X_test)
    #
    # total = len(y_true)
    # right = 0
    #
    # zero = 0
    # one= 0
    # two = 0
    #
    # k = y_pred[0]
    # l = y_true[0]
    #
    # for i in range(0, len(y_pred)):
    #     if (y_pred[i] == 0):
    #         zero += 1
    #     elif (y_pred[i] == 1):
    #         one += 1
    #     else:
    #         two += 1
    # print(zero , one , two)
    #
    #
    # for i in range (len(y_pred)):
    #     if (y_pred[i] == y_true[i]):
    #         right += 1
    # accuracy = float(right)/total * 100
    # print (accuracy)
    #
    # zero = 0
    # one = 0
    # two = 0
    #
    # for i in range(0, len(y_true)):
    #     if (y_true[i] == 0):
    #         zero += 1
    #     elif (y_true[i] == 1):
    #         one += 1
    #     else:
    #         two += 1
    # print(zero , one , two)
    #
    #     # print(classification_report(y_true, y_pred))




# def get_data(lims_ID, ep):

    # # import h5py file for each lims ID video
    # hf = h5py.File(('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\h5 files/training_data_pervectornorm' + str(lims_ID) + '.h5'), 'r')
    # # get a list of all keys in the file dictionary, where in keys correspond to frame number
    # data_tables = hf.keys()
    # # create variable k, which will keep track of current frame number
    # k = 0
    #
    # # important wheel data, get rid of NaN values, and find first non-NaN index. This index will be our starting point
    # # (don't want to train on frames without a wheel value, since it is part of our feature vector)
    # wheel = joblib.load('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\Wheel\dxds'+ str(lims_ID)+'.pkl')
    # first_non_nan = next(x for x in wheel if not isnan(x))
    # first_index = np.where(wheel == first_non_nan)[0]
    # imp = Imputer(missing_values='NaN', strategy='mean')
    #
    # # normalize wheel data according to wheel scaler
    # wheel = imp.fit_transform(wheel)
    #
    #
    # label = 'fidget'
    # index = ep.get_labels().index(label) + 1
    # fidget_vector = np.array(ep.get_per_frame_data()[index])
    #
    # relevant_fidget = 0
    # for i in range (len(fidget_vector)):
    #     if fidget_vector[i] == 1:
    #         relevant_fidget += 1
    #
    # label = 'walking'
    # index = ep.get_labels().index(label) + 1
    # walking = np.array(ep.get_per_frame_data()[index])
    #
    # label = 'running'
    # index = ep.get_labels().index(label) + 1
    # running = np.array(ep.get_per_frame_data()[index])
    #
    # movement_vector = []
    # movement_vector.append([sum(x) for x in zip(walking, running)])
    #
    # # set your desired starting frame relative to first_index
    # start_Frame = 0
    #
    # # store length of feature vectors
    # number_of_features = 5713
    # # set desired frames per training block
    # frames_per_block = 60
    #
    #
    # # update k to fist_index plus any desired start frame offset!!!
    # k = start_Frame + first_index
    #
    # # calculate how many frames the training data consists of
    # if len(data_tables) - start_Frame > 0:
    #     data_length = len(data_tables) - start_Frame
    # else:
    #     data_length = 0
    #
    # #initialize a counter
    # count = 0
    # flex_count = 0
    # rel_count = 0
    # # initialize behavior string
    # beh_type = ""
    #
    # # initialize feature data and label data arrays
    # y_train = []
    # y_track = []
    # feature_data = np.empty((data_length, number_of_features+1), dtype = float)
    # print(feature_data.dtype)
    # neither_count = 0
    # fidget_count = 0
    # movement_count= 0
    #
    #
    # # while we have block_zframes left for each training block, add feature vector to training data and label to training label
    # while k+frames_per_block < len(data_tables):
    #     # get data of first frame
    #     temp = np.array(hf.get('frame number ' + str(int(k))))
    #     # for each frame, add the associated behavior attributes
    #     # beh = mode_beh(beh_present(hf.get('frame number ' + str(int(k+15))).attrs['behavior']))
    #     if fidget_vector[k+30] == 1:
    #         beh = 0
    #     elif movement_vector[0][k+30] == 1:
    #         beh = 2
    #     else:
    #         beh = 1
    #
    #
    #     # Concatenate the feature vectors of the next block size frames
    #     for index in range(k+1, k+frames_per_block):
    #         # temp = np.hstack((temp, np.array(hf.get('frame number ' + str(int(index))))))
    #         try:
    #             temp = ((temp + np.array(hf.get('frame number ' + str(int(index)))))/2)
    #         except:
    #             print(k)
    #             continue
    #         # for each frame, add the associated behavior attributes
    #         # beh_type += beh_present(hf.get('frame number ' + str(int(k))).attrs['behavior'])
    #     # set the corrrespoding row of feature data to concatenated frame features of size block_size
    #
    #     if beh == 1 and neither_count <= relevant_fidget:
    #         feature_data[count, 0:number_of_features] = temp
    #         feature_data[count, number_of_features] = beh
    #         y_train.append(beh)
    #         count += 1
    #         neither_count += 1
    #     elif beh == 0:
    #         feature_data[count, 0:number_of_features] = temp
    #         feature_data[count, number_of_features] = beh
    #         y_train.append(beh)
    #         count += 1
    #         fidget_count += 1
    #     elif beh == 2:
    #         count += 1
    #     # iterate current frame
    #     k += 1
    #     # update how many frames are left to process
    #     data_length = data_length - 1
    #     # based on most frequent behavior in behavior attribute string, add label to training label for training block
    #     # reset behavior attribute for next block
    #     beh_type = ""
    #
    #     if count%1000 == 0:
    #         print(count)




    # return {'feature_data': feature_data, 'y_train': y_train, 'y_track': y_track, 'number': fidget_count+ neither_count + movement_count}

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
    # # '503412730', '497060401','502741583', '501004031', '500860585', '501560436', '503412730', '501773889'
    lims_ID = ['503412730', '497060401','502741583', '501004031', '500860585', '501560436', '503412730', '501773889']
    # # initialize training data and data label arrays, as well a a loop counter
    # y_train = []
    # y_track = []
    # feature_data=[]
    # t= 0
    #
    # # for each limd_ID, get trainign and label data, and combine it with previous lims ID training and label data
    # for itm in lims_ID:
    #     ex = ep("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation", itm)
    #     data = get_data(itm, ex)
    #     if t == 0:
    #         y_train = data['y_train']
    #         y_track = data['y_track']
    #         feature_data = data['feature_data'][0:len(y_train)]
    #         print(itm + ' video done')
    #         print ( data['number'])
    #     else:
    #         y_vector = data['y_train']
    #         y_train = np.concatenate((y_train, y_vector))
    #         y_track = np.concatenate((y_track, data['y_track']))
    #         vector = data['feature_data'][0:len(y_vector)]
    #         feature_data = np.vstack((feature_data, vector))
    #         print(itm + ' video done')
    #
    #     t += 1
    #
    # print('feature processing finished')

    run_svm(lims_ID)