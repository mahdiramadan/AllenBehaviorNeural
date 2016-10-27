"""behaviorAnnotation.py by Mahdi Ramadan, 06-18-2016
This program will be used for video annotation and display.
Referred to as behaviorAnnotation by pycharm IDE
"""
# behaviorAnnotation.py must be in same folder as raw_behavior.py (to avoid adding files to path issues)

from raw_behavior import RawBehavior as rb
from stimulus_behavior import StimulusBehavior as sb
from synced_videos import SyncedVideos as sv
from excel_processing import ExcelProcessing as ep
# from image_processing import ImageProcessing as ip
from wheel_data import WheelData as wd
# from machine_learning import MachineLearning as ml
from lims_database import LimsDatabase as ld
from sync_meta import SyncMeta as sm
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import cv2
import sys
import pandas

class DataAnalysis:
    def __init__(self,exp_folder, lims_ID):
                # set up objects with parameters to be used
                # If class calls on another class that requires a lims_ID,
                # class must have a lims_ID input (add input below)
                self.rb = rb(exp_folder, lims_ID)
                self.sb = sb(exp_folder)
                self.sv = sv(exp_folder, lims_ID)
                self.ep = ep(exp_folder,lims_ID)
                # self.ip = ip(exp_folder, lims_ID)

                # self.ml = ml(exp_folder, lims_ID)
                # self.ld = ld(lims_ID)
                # self.sm = sm(exp_folder, lims_ID)

# Actual running script below

# videos on this laptop stored in "/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos"
# example LIMS ID is 501021421

# input LIMS ID or directory to files of interest!
# RawBehavior, Stimulusbehavior, SyncedVideos, ExcelProcessing take in video directory
# LimsDatabase takes in LIMS ID

# initializes all DataAnalysis objects, takes video dire ctory and lims ID
DataAnalysis = DataAnalysis("C:\Users\mahdir\Desktop\Mahdi files", "500860585")

# data labels for annotations are: "From", "To", "chattering", "trunk_present", "grooming", "trunk_absent", "running"
# "fidget", "tail_relaxed", "tail_tense", "flailing_present", "flailing_absent", "walking", "person", "limsid", "timestamp"

data = DataAnalysis.sv.video_annotation()




















