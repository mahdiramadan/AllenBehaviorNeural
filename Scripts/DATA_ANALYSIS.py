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
import warnings




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

    results_fid = [[]for _ in range(len(stimuli))]
    results_mov = [[] for _ in range(len(stimuli))]
    results_neither = [[] for _ in range(len(stimuli))]

    func = 0

    A_count = 0
    B_count = 0
    C_count = 0


    stim_A_fidget_cux2 = []
    stim_A_fidget_emx1 = []
    stim_A_fidget_nr5a1 = []
    stim_A_fidget_rbp4 = []
    stim_A_fidget_rorb = []
    stim_A_fidget_scnn1a = []

    stim_B_fidget_cux2 = []
    stim_B_fidget_emx1 = []
    stim_B_fidget_nr5a1 = []
    stim_B_fidget_rbp4 = []
    stim_B_fidget_rorb = []
    stim_B_fidget_scnn1a = []

    stim_A_fidget = []
    stim_B_fidget = []
    stim_C_fidget = []

    excel_data = pandas.read_excel('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\global_report.xlsx', sheetname = 'all')
    excel_ids = excel_data['lims ID']
    a= excel_ids[0]


    for id in ids:
        try :
            qc_status = np.array(find_status(id))
            mouse = qc_status[(0)][0].split(';')
            cre = mouse[0].split(',')
            creline = cre[0]
            mouse_id_whole = mouse[2].split('-')
            mouse_id = mouse_id_whole[1]


            stim = qc_status[(0)][3]
            count_t = 0

            if stim == 'three_session_B':
                data = ann_data[count]

                t_fid = 0
                for f in range(len(data)):
                    if data[f] == 0:
                        t_fid += 1

                for m in ids:
                    try:
                        qc_status_t = np.array(find_status(m))
                        mouse_t = qc_status_t[(0)][0].split(';')
                        cre_t = mouse_t[0].split(',')
                        creline_t = cre_t[0]
                        mouse_id_whole_t= mouse_t[2].split('-')
                        mouse_id_t= mouse_id_whole_t[1]
                        stim_t = qc_status_t[(0)][3]

                        if mouse_id == mouse_id_t and stim_t == 'three_session_C' and m != id:
                            data_t = ann_data[count_t]

                            t_fid_t = 0
                            for f in range(len(data_t)):
                                if data_t[f] == 0:
                                    t_fid_t += 1
                            if creline_t == 'Cux2-CreERT2':
                                stim_A_fidget_cux2.append(round((float(t_fid)/len(data))*100, 2))
                                stim_B_fidget_cux2.append(round ((float(t_fid_t)/len(data_t))*100, 2))
                            elif creline_t == 'Emx1-IRES-Cre':
                                stim_A_fidget_emx1.append(round((float(t_fid)/len(data))*100, 2))
                                stim_B_fidget_emx1.append(round ((float(t_fid_t)/len(data_t))*100, 2))
                            elif creline_t == 'Nr5a1-Cre':
                                stim_A_fidget_nr5a1.append(round((float(t_fid)/len(data))*100, 2))
                                stim_B_fidget_nr5a1.append(round ((float(t_fid_t)/len(data_t))*100, 2))
                            elif creline_t == 'Rbp4-Cre':
                                stim_A_fidget_rbp4.append(round((float(t_fid)/len(data))*100, 2))
                                stim_B_fidget_rbp4.append(round ((float(t_fid_t)/len(data_t))*100, 2))
                            elif creline_t == 'Rorb-IRES2-Cre':
                                stim_A_fidget_rorb.append(round((float(t_fid)/len(data))*100, 2))
                                stim_B_fidget_rorb.append(round ((float(t_fid_t)/len(data_t))*100, 2))
                            elif creline_t == 'Scnn1a-Tg3-Cre':
                                stim_A_fidget_scnn1a.append(round((float(t_fid)/len(data))*100, 2))
                                stim_B_fidget_scnn1a.append(round ((float(t_fid_t)/len(data_t))*100, 2))
                            else:
                                print(creline_t)


                        count_t += 1
                    except:
                        count_t += 1
                        continue

            count += 1
        except:
            count +=1
            continue



    all_A = np.concatenate((stim_A_fidget_cux2, stim_A_fidget_emx1, stim_A_fidget_nr5a1, stim_A_fidget_rbp4, stim_A_fidget_rorb, stim_A_fidget_scnn1a))
    all_B = np.concatenate((stim_B_fidget_cux2, stim_B_fidget_emx1, stim_B_fidget_nr5a1, stim_B_fidget_rbp4, stim_B_fidget_rorb, stim_B_fidget_scnn1a))

    plt.title(' Intra-Mouse Stim B versus stim C Fidget rate')
    plt.ylabel('Stimulus C Fidget Rate')
    plt.xlabel('Stimulus B Fidget Rate')
    plt.show(hist2d(all_A, all_B, bins= 18))

    # colors = ['b', 'c', 'y', 'm', 'r', 'k']
    #
    # cux2 = plt.scatter(stim_A_fidget_cux2, stim_B_fidget_cux2, marker='o', color=colors[0])
    # emx1 = plt.scatter(stim_A_fidget_emx1, stim_B_fidget_emx1, marker='o', color=colors[1])
    # nr5a1 = plt.scatter(stim_A_fidget_nr5a1, stim_B_fidget_nr5a1, marker='o', color=colors[2])
    # rbp4 = plt.scatter(stim_A_fidget_rbp4, stim_B_fidget_rbp4, marker='o', color=colors[3])
    # rorb = plt.scatter(stim_A_fidget_rorb, stim_B_fidget_rorb, marker='o', color=colors[4])
    # scnn1a = plt.scatter(stim_A_fidget_scnn1a, stim_B_fidget_scnn1a, marker='o', color=colors[5])
    #
    # plt.legend((cux2, emx1, nr5a1, rbp4, rorb, scnn1a),
    #            ('Cux2', 'Emx1', 'Nr5a1', 'Rbp4', 'Rorb', 'Scnn1a'),
    #            scatterpoints=1,
    #            loc='lower right',
    #            ncol=3,
    #            fontsize=8)
    #
    # plt.title(' Intra-Mouse Stim B versus stim C Fidget rate')
    # plt.ylabel('Stimulus B Fidget Rate')
    # plt.xlabel('Stimulus C Fidget Rate')
    # plt.show()

    # workbook = xlsxwriter.Workbook('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\A_cre.xlsx')
    # worksheet = workbook.add_worksheet()
    #
    #
    # worksheet.write_row(11, 0, stim_A_fidget_scnn1a)
    # # worksheet.write_row(12, 0, stim_B_fidget_scnn1a)
    # workbook.close()


    #     if stim == 'three_session_A':
    #         A_count += 1
    #     if stim == 'three_session_B':
    #         B_count += 1
    #     if stim == 'three_session_C':
    #         C_count += 1
    #
    #
    # time_A = [[] for _ in range(A_count)]
    # time_B = [[] for _ in range(B_count)]
    # time_C = [[] for _ in range(C_count)]
    #
    # A_count = 0
    # B_count = 0
    # C_count = 0

    # for id in ids:
    #
    #     qc_status = np.array(find_status(id))
    #     status = qc_status[(0)][2]
    #
    #     # if status is not None and 'failed' not in status:
    #
    #
    #     func += 1
    #     stimulus = qc_status[(0)][3]
    #     data = ann_data[count]
    #     count += 1
    #
    #
    #     t_fid = 0
    #     for f in range(len(data)):
    #         if data[f] == 0:
    #             t_fid += 1
    #
    #     t_neither = 0
    #     for f in range(len(data)):
    #         if data[f] == 1:
    #             t_neither += 1
    #
    #     t_mov = 0
    #     for f in range(len(data)):
    #         if data[f] == 2:
    #             t_mov += 1
    #
    #     base = (float(t_fid)/len(data))*100
    #
    #     block = []
    #     for i in range (0, len(data), 1000):
    #         p = 0
    #         for k in range (i, i+1000):
    #             try:
    #                 if data[k] == 0:
    #                     p += 1
    #             except:
    #                 continue
    #         block.append(round((float(p/float(1000))*100)/base, 2))
    #
    #     data = block
    #
    #
    #     if stimulus == 'three_session_A':
    #
    #         time_A[A_count] = data
    #         A_count += 1
    #
    #     if stimulus == 'three_session_B':
    #
    #         time_B[B_count] = data
    #         B_count += 1
    #
    #     if stimulus == 'three_session_C':
    #         time_C[C_count] = data
    #         C_count += 1
    #
    #
    # # print np.apply_along_axis(myfunction, axis=1, arr=time_A)
    # # print np.apply_along_axis(myfunction, axis=1, arr=time_B)
    # # print np.apply_along_axis(myfunction, axis=1, arr=time_C)
    #
    #         # Create a new workbook and add a worksheet
    #
    #
    # workbook = xlsxwriter.Workbook('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\.xlsx')
    # worksheet = workbook.add_worksheet()
    #
    # for row, data in enumerate(time_C):
    #     worksheet.write_row(row, 0, data)
    # workbook.close()
    #

def myplot(x, y, nb=32, xsize=500, ysize=500):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    x0 = (xmin+xmax)/2.
    y0 = (ymin+ymax)/2.

    pos = np.zeros([3, len(x)])
    pos[0,:] = x
    pos[1,:] = y
    w = np.ones(len(x))

    P = sph.Particles(pos, w, nb=nb)
    S = sph.Scene(P)
    S.update_camera(r='infinity', x=x0, y=y0, z=0,
                    xsize=xsize, ysize=ysize)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()
    extent = R.get_extent()
    for i, j in zip(xrange(4), [x0,x0,y0,y0]):
        extent[i] += j
    print extent
    return img, extent

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