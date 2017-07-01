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
from scipy.misc import toimage
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
from sync_meta import SyncMeta as sm
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
from Sync_Camera_Stimulus_Analysis import Get_Wheel as gw
from PIL import Image




def run (ids , ann_data):

    # boc = BrainObservatoryCache(manifest_file='boc/manifest.json')


    # Get data container information
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

    # find container status
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

    # read in global report for analysis and read excel ids
    excel_data = pandas.read_excel('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\global_report.xlsx', sheetname = 'all')
    excel_ids = excel_data['lims ID']
    mid_list = excel_data['lims_database_specimen_id']

    # unique = np.unique(rig_list)
    # # unique_operator = np.unique(operator_list).tolist()
    # session_results = [[] for _ in range(17)]


    anxiety = []
    session = []

    for id in ids:
        if id.strip() == '509292861':
            id = id.strip()
            # qc_status = np.array(find_status(id.strip()))
            # status = qc_status[(0)][2]
            # check if failed
            # if 'failed' not in status:
            #     print (' experiment ' + str(id) + ' did not pass QC')
            #

            # get new video directory
            directory = ld(id).get_video_directory()
            temp = directory.split('cns\\')
            new_dir = '\\\\allen\programs\\braintv\production\\' + temp[1]




            nwb_path = None
            h5_file= None
            pkl_file = None
            video_file= None
            # get experiment's associated files
            for file in os.listdir(new_dir):
                if file.endswith(".nwb"):
                    # make sure file is in there!
                    nwb_path = os.path.join(new_dir, file)
                    # input file path, r is for read only
            if bool(nwb_path) == False:
                print('NWB file not found')
                continue

            for file in os.listdir(new_dir):
                # looks for the h5 file and makes the directory to it
                if file.endswith("sync.h5"):
                    h5_file = os.path.join(new_dir, file)
            if bool(h5_file) == False:
                print('H5 file not found')
                continue


            for file in os.listdir(new_dir):
                # looks for the pkl file and makes the directory to it
                if file.endswith("stim.pkl"):
                    pkl_file = os.path.join(new_dir, file)
            if bool(pkl_file) == False:
                print('PKL file not found')
                continue


            # get wheel data and first non Nan value
            grab_wheel = gw(h5_file)
            frames = grab_wheel.return_frames()
            wheel = grab_wheel.getRunningData(pkl_file, frames)
            first_non_nan = next(x for x in wheel if not isnan(x))
            first_index = np.where(wheel == first_non_nan)[0]
            imp = Imputer(missing_values='NaN', strategy='mean')
            # normalize wheel data according to wheel scaler
            wheel = imp.fit_transform(wheel)
            k = first_index[0]

            # get behavior and neural activity timing, as well as annotated data and dff traces
            neuron = BrainObservatoryNwbDataSet(nwb_path)
            n_data = neuron.get_dff_traces()

            beh_time = sm(new_dir).get_frame_times_behavior()
            neuro_time = sm(new_dir).get_frame_times_physio()
            data = ann_data[count]

            # get visual cortex movie
            path = '\\\\allen\\programs\\braintv\\production\\neuralcoding\\prod6\\specimen_501800347\\ophys_experiment_509292861\\processed\\concat_31Hz_0.h5'
            F = h5py.File(path)

            # determine how many unique fidget examples there are
            flu_count = 0

            limit = len(data)-200
            if limit > len(beh_time):
                limit = len(beh_time)
            if limit > len(neuro_time):
                limit = len(neuro_time)

            for f in range(len(data)-200):
                if (data[f] == 1 and data[f + 1] == 0) or (data[f] == 2 and data[f + 1] == 0):
                    flu_count += 1

            # initialize data arrays
            fluorescent_traces = [[[] for _ in range (len(n_data[1]))] for _ in range(flu_count)]
            fluorescent_traces_cell = [[[] for _ in range(flu_count)] for _ in range(len(n_data[1]))]
            video_traces = [[[] for _ in range (flu_count)] for _ in range(300)]

            print(flu_count)
            flu_count = 0

            indices = []



            print('calculating average movie and neural activity')
            # for each frame, check whether its fidget or not
            for f in range(len(data)-200):
                if (data[f] == 1 and data[f+1] == 0) or (data[f] == 2 and data[f+1] == 0):
                    # get behavior time (must be offset by the index of first wheel value since annotated data starts then)
                    b_time = beh_time[f+k]
                    # use to get associated video time in seconds
                    t = beh_time[f]
                    # get mapped fluorescence index
                    idx = (np.abs(neuro_time - b_time)).argmin()
                    # store each cell's fluorescent trace 100 frames back and 200 forward from start of fidget
                    for p in range (len(n_data[1])):
                        # keeps track per fidget example
                        fluorescent_traces[flu_count][p] = n_data[1][p][idx-100:(idx+200)]
                        # keeps track per cell
                        fluorescent_traces_cell[p][flu_count] = n_data[1][p][idx - 100:(idx + 200)]
                    # save video trace of neural activity
                    # for h in range(300):
                    #
                    #     video_traces[h][flu_count] = np.array(F['data'][ int(idx-100+h),:,:])


                    # data_pointer.release()
                    # cv2.destroyAllWindows()
                    flu_count += 1



                if f  %1000 == 0:
                    print(f)


            average_video = [[] for _ in range(300)]
            firing1 = []
            firing2 = []
            indices = []

            # for each fidget example, get examples where there is a significant difference between during and after fidget
            # response, average those videos together and write them out (then use makeavi programto make video)
            # for i in range (flu_count):
            #     temp = np.array(fluorescent_traces[i])
            #     array1 = np.mean(temp[:, 100:150])
            #     array2 = np.mean(temp[:, 150:200])
            #
            #     # if array2 > 0.08 and array1 < 0.08:
            #     if array2-array1 > 0.06:
            #         indices.append(i)
            #         firing1.append(array1)
            #         firing2.append(array2)
            # temp =[]
            # for p in range(300):
            #     temp= [video_traces[p][i] for i in indices]
            #     average_video[p] = np.mean(temp, axis=0)
            #     # print(np.min(video_traces[p][0]), np.max(video_traces[p][0]))
            #     toimage(average_video[p], cmin=0, cmax=400).save(
            #         'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\neural_images_select2\\' + 'image_' + str(
            #             p) + '.png')


            # Use code to plot average fluorescence per fidget example

            # # plt.plot(firing, color = 'b')
            # plt.title("Average Fluroscence per Fidget Example During and After Fidget")
            # plt.plot(firing1, 'bo')
            # plt.plot(firing2, 'ro')
            # plt.xlabel('Fidget Example')
            # plt.ylabel('Average Fluorescence')
            # plt.show()
            #
            # workbook = xlsxwriter.Workbook(
            #     'C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\average_fluoresence_select2.xlsx')
            # worksheet = workbook.add_worksheet()
            #
            # worksheet.write_row(0, 0, firing1)
            # worksheet.write_row(1, 0, firing2)
            #
            # workbook.close()


            # use code to plot average cell response

            fluorescence_average = [[] for _ in range(len(n_data[1]))]

            plt.title('Fluorescent Traces During Fidget for id ' + str(id))


            for p in range(len(n_data[1])):
                fluorescence_average[p] = np.mean(fluorescent_traces_cell[p][:], axis = 0)
                fluorescence_average[p] = fluorescence_average[p] - fluorescence_average[p][0]
                if np.max(fluorescence_average[p]) > 20:
                    print(p)
                # plt.plot(fluorescence_average[p])

            # plt.show()

            plt.imshow(fluorescence_average[:])
            plt.clim(-0.01, 0.01)
            plt.show()
        count += 1
        # except:
        #     count += 1
        #     continue




def array2image(a):
    if a.typecode() == Numeric.UnsignedInt8:
        mode = "L"
    elif a.typecode() == Numeric.Float32:
        mode = "F"
    else:
        raise ValueError, "unsupported image mode"
    return Image.fromstring(mode, (a.shape[1], a.shape[0]), a.tostring())

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

            if ((float(f) / len(data[i]))*100) > 50:
                print(i)
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
    # get list of lims id
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    text_file = open("C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\\LIMS_IDS.txt", "r")
    lims_ids = text_file.read().split(',')
    # print (lims_ids[79])

    # get annotated data
    ann_data = get_all_data(lims_ids)

    # use code to create behavior frequency histograms
    # create_histogram(ann_data)
    #
    # run main analysis script
    run(lims_ids, ann_data)