"""behaviorAnnotation.py by Mahdi Ramadan, 06-18-2016
This program will be used for video annotation and display.
Referred to as behaviorAnnotation by pycharm IDE
"""
# behaviorAnnotation.py must be in same folder as raw_behavior.py (to avoid adding files to path issues)

from Machine_Learning_Model import AnnotationModel as am
from Sync_Camera_Stimulus import Get_Wheel as gw
from lims_database import LimsDatabase as ld
from batch_ophyse import BatchOphys as bo
from synced_videos import SyncedVideos as sv
# from machine_learning import MachineLearning as ml

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import cv2
import sys
import pandas

from psycopg2 import connect


class DataAnalysis:
    def __init__(self, lims_ID):
        self.ld = ld(lims_ID)
        self.sv = sv ( lims_ID )
# Actual running script below

# videos on this laptop stored in "/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos"
# example LIMS ID is 501021421

# input LIMS ID or directory to files of interest!
# RawBehavior, Stimulusbehavior, SyncedVideos, ExcelProcessing take in video directory
# LimsDatabase takes in LIMS ID

# IDs = bo().get_all_ophys ('2016-3-01' , '2016-06-30')
#
# with open('C:\Users\mahdir\Documents\Allen Projects\Behavior Annotation\LIMS_IDS.txt', 'w') as file_out:
#     # if item == 'true':
#     #     print('Feature Selection on')
#     #     file_out.write ('Feature Selection on \\\\n')
#     file_out.write( str(IDs) )

# initializes all DataAnalysis objects, takes video dire ctory and lims ID
IDs = ['510660713']

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


for Lims_ID in IDs:

    qc_status = np.array(find_status (Lims_ID))
    status = qc_status[(0)][1]

    if 'published' not in status:
        print (' experiment ' + str(Lims_ID) + ' did not pass QC')
        break

    DataAnalysis = DataAnalysis(Lims_ID)

    # data labels for annotations are: "From", "To", "chattering", "trunk_present", "grooming", "trunk_absent", "running"
    # "fidget", "tail_relaxed", "tail_tense", "flailing_present", "flailing_absent", "walking", "person", "limsid", "timestamp"

    video_directory = DataAnalysis.ld.get_video_directory()

    for file in os.listdir(video_directory):
        # looks for the h5 file and makes the directory to it
        if file.endswith("sync.h5") and file.startswith(Lims_ID):
            h5_file = os.path.join(video_directory, file)
    if bool(h5_file) == False:
        print('H5 file not found')

    for file in os.listdir(video_directory):
        # looks for the pkl file and makes the directory to it
        if file.endswith("stim.pkl") and file.startswith(Lims_ID):
            pkl_file = os.path.join(video_directory, file)
    if bool(pkl_file) == False:
        print('H5 file not found')

    for file in os.listdir(video_directory):
        # looks for the pkl file and makes the directory to it
        if file.endswith("-0.avi") and file.startswith(Lims_ID):
            video_file = os.path.join(video_directory, file)
    if bool(video_file) == False:
        print('Movie file not found')

    video_pointer = cv2.VideoCapture(video_file)
    # set video pointer to first frame, and read first frame
    video_pointer.set(1, 1000)

    x1 = 180
    x2 = 360
    y1 = 210
    y2 = 350

    ret, frame = video_pointer.read()


    # crops and converts frame into desired format
    frame = cv2.cvtColor(frame[x1:x2, y1:y2], cv2.COLOR_BGR2GRAY)

    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Get wheel data
    grab_wheel = gw (h5_file)
    frames = grab_wheel.return_frames()
    wheel = grab_wheel.getRunningData(pkl_file, frames)

    if (np.max(wheel) > 20):
        print('ABNORMAL WHEEL DATA')

    # Predict Behavior
    am (Lims_ID, video_file, wheel, x1, x2, y1, y2)

    # Save behavior video
    DataAnalysis.sv.video_annotation(video_file, Lims_ID, wheel)




















