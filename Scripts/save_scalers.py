"""save_scalers.py by Mahdi Ramadan, 07-12-2016
This program will save feature scalers for use in training and testing sets
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
    hf = h5py.File((exp_folder + '\h5 files\\'+ 'training_data_scaling' + str(lims_ID) + '.h5'), 'r')
    group = hf.get('scaler data')

    # set wheel data scaler
    dir = os.path.join(exp_folder + '\Wheel\\', 'dxds' + str(lims_ID) + '.pkl')
    wheel = joblib.load(dir)
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]
    k = first_index[0]
    imp = Imputer(missing_values='NaN', strategy='mean')
    wheel = imp.fit_transform(wheel)
    min_max_scaler_wheel = preprocessing.MinMaxScaler(feature_range = (-1,1))
    mim_max_scaler = min_max_scaler_wheel.fit_transform(wheel)
    joblib.dump(min_max_scaler_wheel, exp_folder + '\Scalers\\' + 'wheel_scale_' + str(lims_ID) + '.pkl' )

    # create scaler for frame data
    frame_data = np.array(group.get('features'))[:, 1:1905]
    min_max_scaler_frames = preprocessing.MinMaxScaler(feature_range = (-1,1))
    mim_max_scaler_frames = min_max_scaler_frames.fit_transform(frame_data)
    joblib.dump(min_max_scaler_frames, exp_folder + '\Scalers\\' + 'frame_scale_' + str(lims_ID) + '.pkl')


    # create scaler for optical flow
    optical_data = np.array(group.get('features'))[:, 1905:3809]
    min_max_scaler_optical = preprocessing.MinMaxScaler(feature_range = (-1,1))
    mim_max_scaler_optical = min_max_scaler_optical.fit_transform(optical_data)
    joblib.dump(min_max_scaler_optical, exp_folder + '\Scalers\\' + 'optical_scale_' + str(lims_ID) + '.pkl')

    # set scaler for optical angle
    angle_data = np.array(group.get('features'))[:, 3809:5713]
    min_max_scaler_angle = preprocessing.MinMaxScaler(feature_range = (-1,1))
    mim_max_scaler_angle = min_max_scaler_angle.fit_transform(angle_data)
    joblib.dump(min_max_scaler_angle, exp_folder + '\Scalers\\' + 'angle_scale_' + str(lims_ID) + '.pkl')


if __name__ == '__main__':
    # initializes code with path to folder and lims ID
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    exp_folder = 'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation'
    lims_ID = '501560436'

    # run_whole_video(exp_folder, lims_ID)

    p = Process(target=run_whole_video(exp_folder, lims_ID), args= (exp_folder, lims_ID))
    p.start()
    p.join()