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

    QUERY = "SELECT ec.id, ec.workflow_state, eso.stimulus_name, eso.id, eso.workflow_state FROM experiment_sessions eso LEFT JOIN experiment_containers ec ON ec.id = eso.experiment_container_id WHERE eso.id='{}';"

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

    results_fid = [[]for _ in range(len(stimuli))]
    results_mov = [[] for _ in range(len(stimuli))]
    results_neither = [[] for _ in range(len(stimuli))]

    func = 0
    print( len(ids))

    for id in ids:

        qc_status = np.array(find_status(id))
        status = qc_status[(0)][1]

        if status is not None and 'failed' not in status:

            # get nwb file data

            nwb_file = open_nwb(id)
            if not nwb_file:
                count += 1
                continue

            func += 1
            stimulus = qc_status[(0)][2]

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

            # open data to presentation branch
            visual = nwb_file['stimulus']['presentation']

            # iterate over stimulus types, if type is found in data, then get frame durations
            for stim in stimuli:
                if stim in visual:

                    # get unique frame numbers
                    try:
                        frames = np.unique([nwb_file['stimulus']['presentation'][stim]['data']])
                    except:
                        print( 'could not access stimulus timing files for id ' + str(id))
                    num_stim = []

                    # to get continuous frame ranges, un-comment code right below
                    # ranges = []
                    # for k, g in groupby(enumerate(frames), lambda (i, x): i - x):
                    #     group = map(itemgetter(1), g)
                    #     ranges.append((group[0], group[-1]))
                    a= 0
                    b = 0
                    c = 0

                    if len(frames) > 1:
                        for i in range(0, len(frames)-1, 2):
                            # If frame ranges is of length less than 150, then assume it is giving frame ranges. Otherwise,
                            # assume it is giving individual frames ( range e.g. 0 - 10000,
                            # individual e.g. 0, 1, 2, 3.... 10000)
                            # shortest stimulus is 30 seconds, longest movie is 3800 seconds, 3800/30 is about 150
                            # Thus, you wil never have more than 150 discrete frame ranges
                            if math.isnan(frames[i]) or math.isnan(frames[i+1]) or frames[i+1] < 0 or frames[i] < 0:
                                # print('NaN values for Stimulus frames for ' + str(id) + ' with stimulus ' + str(stim))
                                continue
                            else:
                                num_stim.append(frames[i+1] - frames[i])
                                for k in range(int(frames[i]), int(frames[i+1])):
                                    try:
                                        if data[int(k)] == 0:
                                            a += 1
                                        elif data[int(k)] == 1:
                                            b += 1
                                        else:
                                            c+= 1

                                    except:
                                        continue


                    elif len(frames) <= 1:
                        print ( ' abnormal frame count for stimulus for id ' + str(id) + ' for stimulus ' + str(stim))
                        continue


                # if type if not in data, take next type
                else:
                    continue

                # if stimulus == 'Stimulus'
                try:
                    if ((a/float(np.sum(num_stim)))*100) <= 100 and ((a/float(np.sum(num_stim)))*100) != 0:
                        results_fid[int(stimuli.index(stim))].append((a/float(np.sum(num_stim)))*100)

                except:

                    continue

                try:
                    if ((b/ float(np.sum(num_stim))) * 100) <= 100 and ((b/ float(np.sum(num_stim))) * 100) != 0:
                        results_neither[int(stimuli.index(stim))].append((b / float(np.sum(num_stim))) * 100)

                except:

                    continue

                try:
                    if ((c/ float(np.sum(num_stim))) * 100) <= 100 and ((c/ float(np.sum(num_stim))) * 100) != 0:
                        results_mov[int(stimuli.index(stim))].append((c / float(np.sum(num_stim))) * 100)

                except:

                    continue
                # print (str(stim) + ' has ' + str(sum) + str(label))
            count += 1

        else:
            count += 1
    print(func)


    for stim in stimuli:
        common_params = dict(bins=20,
                             range=(0, 100),
                             normed=False
                             )
        print( str(stim))
        print (str(np.mean(results_fid[int(stimuli.index(stim))])) + ' ' + str(np.std(results_fid[int(stimuli.index(stim))])))
        print (str(np.mean(results_mov[int(stimuli.index(stim))])) + ' ' + str(np.std(results_mov[int(stimuli.index(stim))])))
        print (str(np.mean(results_neither[int(stimuli.index(stim))])) + ' ' + str(np.std(results_neither[int(stimuli.index(stim))])))
        print ()

        plt.hist((results_fid[int(stimuli.index(stim))], results_mov[int(stimuli.index(stim))], results_neither[int(stimuli.index(stim))]),
                 label=('Fidget Frequency', 'Movement Frequency', 'Neither frequency'), **common_params)
        plt.legend(loc='upper right')
        plt.xlabel(' Ratio of ' + str(stim) + ' Displaying Behavior')
        plt.ylabel(' Number of Experiments ')
        plt.title('Distribution of Behavior Frequency During '  + str(stim) )
        plt.show()


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
    text_file = open("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\LIMS_IDS.txt", "r")
    lims_ids = text_file.read().split(',')
    # print (lims_ids[5] + lims_ids[12] + lims_ids[14] + lims_ids[25] +lims_ids[34] + lims_ids[41] +lims_ids[46] + lims_ids[49]  +lims_ids[54] + lims_ids[55]   )
    ann_data = get_all_data(lims_ids)
    # create_histogram(ann_data)
    #
    run(lims_ids, ann_data)