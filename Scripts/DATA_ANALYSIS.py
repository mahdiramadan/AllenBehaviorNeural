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
import xlsxwriter
from math import log
from scipy import stats
from skimage.feature import hog
from skimage import data, color, exposure
from scipy.optimize import curve_fit
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import neighbors, datasets
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
import warnings
from datetime import datetime
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from matplotlib import cm as CM
from matplotlib import mlab as ML



def run (ids , ann_data):
    # create CAM stimulus definition visual
    count = 0

    db_params = {
        u'dbname': u'lims2',
        u'user': u'limsreader',
        u'host': u'limsdb2',
        u'password': u'limsro',
        u'port': 5432
    }

    QUERY = " ".join(("SELECT sp.name, ec.id, ec.workflow_state, eso.stimulus_name, eso.id, eso.workflow_state",
                      "FROM experiment_sessions eso",
                      "LEFT JOIN experiment_containers ec ON ec.id = eso.experiment_container_id",
                      "JOIN specimens sp ON sp.id=eso.specimen_id",
                      "WHERE eso.id='{}';"))

    def get_db_cursor(dbname, user, host, password, port):
        con = connect(dbname=dbname,
                      user=user,
                      host=host,
                      password=password,
                      port=port)
        return con.cursor()

    cur = get_db_cursor(**db_params)

    def find_status(lims_id):
        query = QUERY.format(lims_id)
        cur.execute(query)
        return cur.fetchall()


    # types of possible stimuli
    stimuli = ['spontaneous_stimulus', 'drifting_gratings_stimulus', 'natural_movie_one_stimulus',
               'natural_movie_two_stimulus',
               'natural_movie_three_stimulus', 'static_gratings_stimulus',
               'locally_sparse_noise_stimulus']

    A_count = 0
    B_count = 0
    C_count = 0

    excel_data = pandas.read_excel('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\global_report.xlsx', sheetname = 'all')
    excel_ids = excel_data['lims ID']
    mid_list = excel_data['lims_database_specimen_id']

    # unique = np.unique(rig_list)
    # # unique_operator = np.unique(operator_list).tolist()
    # session_results = [[] for _ in range(17)]

    anxiety = []
    session = []

    for id in ids:

        data = ann_data[count]

        t_fid = 0
        for f in range(len(data)):
            if data[f] == 0:
                t_fid += 1

        t_neither = 0
        for f in range(len(data)):
            if data[f] == 1:
                t_neither += 1

        t_mov = 0
        for f in range(len(data)):
            if data[f] == 2:
                t_mov += 1


        for index, itm in enumerate(excel_ids):
            if str(itm).strip() == id.strip():
                track = index


        temp = []
        if track is not empty:
            MID = int(mid_list[track])
            c = 1
            for i in range(0,track):
                if int(mid_list[i]) == MID:
                    temp.append(i)
                    c += 1
            session_results[c-1].append(round((float(t_fid)/len(data))*100, 2))
            # session.append(c)

        count += 1

    # plt.subplot(111)
    # gridsize = 20
    #
    # plt.hexbin(anxiety, session, gridsize=gridsize, cmap=CM.jet, bins=None)
    # plt.axis([np.min(anxiety), np.max(anxiety), np.min(session), np.max(session)])
    #
    # cb = plt.colorbar()
    # cb.set_label('mean value')
    # plt.title('Effect of Number of Mouse Sessions on Fidget Rate Heat Map')
    # plt.xlabel('Fidget Rate')
    # plt.ylabel('Number of  Sessions')
    # plt.show()


    workbook = xlsxwriter.Workbook('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\session_average_results.xlsx')
    worksheet = workbook.add_worksheet()

    for indx, data in enumerate(session_results):
        worksheet.write_row(indx, 0, data)

    workbook.close()



def myfunction (x):
    return len (x)

def open_nwb(lims_ID):

    directory = ld(lims_ID.strip()).get_video_directory()
    nwb_file = False
    # opens nwb file
    for file in os.listdir(directory):
        if file.endswith("nwb"):
            # make sure file is in there!
            nwb_path = os.path.join(directory, file)
            # input file path, r is for read only
            nwb_file = h5py.File(nwb_path, "r")

    return nwb_file



def create_histogram (data):

    values_fidget = []
    values_movement = []
    values_neither = []


    for i in range(len(data)):
        f = 0
        m = 0
        n = 0
        for k in range(len(data[i])):
            if data[i][k] == 0:
                f += 1
            elif data[i][k] == 1:
                n += 1
            elif data[i][k] == 2:
                m += 1
        try:
            values_fidget.append((float(f) / len(data[i]))*100)
            values_movement.append((float(m) / len(data[i]))*100)
            values_neither.append((float(n) / len(data[i]))*100)

            # if ((float(m) / len(data[i]))*100) < 2:
            #     print(i)
        except:
            continue

    common_params = dict(bins=20,
                         range=(0, 100),
                         normed= False
                         )
    plt.hist((values_fidget, values_movement, values_neither), label= ('Fidget Frequency', 'Movement Frequency', 'Neither frequency'), **common_params)
    plt.legend ( loc = 'upper right')
    plt.xlabel (' Percent of Frames Displaying Behavior ')
    plt.ylabel( ' Number of Experiments ')
    plt.title("Distribution of Behavior Frequency of Occurrence")

    mean = np.mean(values_fidget)
    variance = np.var(values_fidget)
    sigma = np.sqrt(variance)
    x = np.linspace(min(values_fidget), max(values_fidget), 100)
    plt.plot(x, 500*mlab.normpdf(x, mean, sigma), color = ('b') )

    mean = np.mean(values_neither)
    variance = np.var(values_neither)
    sigma = np.sqrt(variance)
    x = np.linspace(min(values_neither), max(values_neither), 100)
    plt.plot(x, 500 * mlab.normpdf(x, mean, sigma), color=('r'))


    plt.show()



def fitExponent(tList,yList,ySS=0):
   '''
   This function finds a
       tList in sec
       yList - measurements
       ySS - the steady state value of y
   returns
       amplitude of exponent
       tau - the time constant
   '''
   bList = [log(max(y-ySS,1e-6)) for y in yList]
   b = matrix(bList).T
   rows = [ [1,t] for t in tList]
   A = matrix(rows)
   #w = (pinv(A)*b)
   (w,residuals,rank,sing_vals) = lstsq(A,b)
   tau = -1.0/w[1,0]
   amplitude = exp(w[0,0])
   return (amplitude,tau)




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

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    text_file = open("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\LIMS_IDS.txt", "r")
    lims_ids = text_file.read().split(',')
    # print (lims_ids[5] + lims_ids[12] + lims_ids[14] + lims_ids[25] +lims_ids[34] + lims_ids[41] +lims_ids[46] + lims_ids[49]  +lims_ids[54] + lims_ids[55]   )
    ann_data = get_all_data(lims_ids)
    # create_histogram(ann_data)
    #

    run(lims_ids, ann_data)