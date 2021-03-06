"""excel_processing.py by Mahdi Ramadan, 06-18-2016
This program will be used for excel file processing of
annotated behavior videos
"""
import os
import pandas
import sys
from raw_behavior import RawBehavior as rb
from synced_videos import SyncedVideos as sv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
from operator import itemgetter
from itertools import groupby
import matplotlib.patches as mpatches



class ExcelProcessing:
    def __init__(self, exl_folder, lims_ID):

        for file in os.listdir(exl_folder+ '\Excel\\'):
            # looks for the excel file and makes the directory to it
            if file.endswith(".xlsx") and file.startswith(lims_ID):
                self.directory = exl_folder +'\Excel\\'
                self.file_string = os.path.join(exl_folder + '\Excel\\', file)
                self.lims_ID = lims_ID
                # excel uploads as a DataFrame type
                # IMPORTANT: zero based excel indexing starts with first row of numbered DATA, not the column labels.
                # This means row zero is the first row of actual Data, not the column labels
                self.data = pandas.read_excel(self.file_string)
                # get video directory and name
                file_name = rb(exl_folder,lims_ID).get_file_string()
                # data pointer to the behavior video annotated
                self.data_pointer = cv2.VideoCapture(file_name)

            else:
                continue

        self.data_present = os.path.isfile(self.file_string)
        self.behavior_data_flow = rb

    def data_valid(self):
        return self.data_present

    def frames_continuous(self):
        # this method checks to see if the labeled frames are continuous. Make sure the labeled frame data is continuous
        # for the rest of the code to work!
        # gets the to and from frames
        To_frames = self.get_to()
        From_frames = self.get_from()

        # for the each iteration, check whether the "to" frame is equal to the "from" frame in the next row
        # if not continuous, returns which rows are discontinuous
        for k in range(len(From_frames)-1):
            if To_frames[k] != From_frames[k+1]:
                return "Frames are not continuous between row number %r and %r of the data" % (k+2, k+3)
            else:
                continue
        return "Frames are continuous!"



    def get_column(self, label):
        # method to extract column of data based on label
        data = self.data[label]
        return data

    def get_categories(self):
        # returns data labels
        categories = list(self.data.columns.values)
        return categories

    def get_size(self):
        # returns size of data (ignoring labels in first row)
        ID_length = len(self.get_id())
        column_length = len(self.get_categories())
        return (ID_length, column_length)

    def get_from(self):
        # returns "from" frames data
        f = self.data['From']
        return f

    def get_to(self):
        # returns "to" frames data
        f = self.data['To']
        return f

    def get_true_false(self, label, index):
        data = self.get_column(label)
        if data[index] == 0:
            return 0
        if data[index] == 1:
            return 1

    def get_zero_frames_range(self, label):
        # returns all the frame ranges that the label specified was equal to zero
        data = self.get_column(label)
        count = 0
        frames = [[], []]
        # Note: first column is all the From frames, second is all the To frames
        for i in data:
            if i == 0:
                frames[0].append(self.get_from()[count])
                frames[1].append(self.get_to()[count])
                count += 1

            else:
                count += 1
                continue

        return frames

    def get_one_frames_range(self, label):
        # returns all the frame ranges that the label specified was equal to one
        data = self.get_column(label)
        count = 0
        frames = [[], []]
        # NOTE: first column is all the From frames, second is all the To frames
        for i in data:
            if i == 1:
                frames[0].append(self.get_from()[count])
                frames[1].append(self.get_to()[count])
                count += 1

            else:
                count += 1
                continue

        return frames

    def get_frame_start(self,count):
        data = self.get_column("From")[count]
        return data

    def get_frame_end(self,count):
        data = self.get_column("To")[count]
        return data

    def video_annotation_labels(self):

        # outputs a .mp4 video with frame number and labeled annotation text

        # gets video file information
        fps = self.data_pointer.get(cv2.cv.CV_CAP_PROP_FPS)
        nFrames = int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frameWidth = int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        frameHeight = int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
        # change 3rd parameter of out function for different playback speeds
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frameWidth, frameHeight))
        ret, frame = self.data_pointer.read()

        # gets the data table with frame number and 0 or 1 for each column label
        frame_data = self.get_per_frame_data()
        # iterates through each row
        for i in range(self.get_first_frame(), self.get_last_frame()+1):
            # prints frame number on frame
            cv2.putText(img=frame,
                        text=str(int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))),
                        org=(20, 100),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0, 0, 0),
                        thickness=2,
                        lineType=cv2.CV_AA)
            count2 = 0
            # iterates for each column label
            for k in range(len(self.get_labels())):
                # if true (value = 1), then we print label
                if frame_data[k+1][i] == 1:
                    count2 += 1
                    # # prints text in green or red depending on column label (optional for color effects)
                    # if k == 0 or k == 3 or k == 6:
                    # color codes of format (x,y,z)
                    #     c = (0,0,255)
                    # else:
                    #     c = (0,255,0)
                    if count2 == 1:
                        cv2.putText(img=frame,
                                    text=str(self.get_labels()[k]),
                                    org=(0 + count2 * 80, 100),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.35,
                                    color=(0, 0, 0),
                                    thickness=1,
                                    lineType=cv2.CV_AA)
                    else:
                        cv2.putText(img=frame,
                                    text=str(self.get_labels()[k]),
                                    org=(0+count2*90, 100),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.35,
                                    color= (0,255,0),
                                    thickness=1,
                                    lineType=cv2.CV_AA)
                else:
                    continue

            # write out the frame
            out.write(frame)
            # read next frame
            ret, frame = self.data_pointer.read()

        # if number of labeled frames is less than number of video frames, just print frame number
        while ret:
            cv2.putText(img=frame,
                        text=str(int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))),
                        org=(20, 100),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.CV_AA)
            out.write(frame)
            # read next frame
            ret, frame = self.data_pointer.read()



    def get_per_frame_data(self):
        # This method takes in the annotated excel data with frame ranges, and returns a data matrix of
        # each frame number annotated, along with the annotation scheme of each label (0 vs. 1) at that frame

        # initiates data lists
        frame_start = []
        frame_end = []
        labels = self.get_labels()
        # initiates the labels x number of frames data list, the minus 3 is to ignore the columns of
        # name, mousid and date
        frame_data = [[] for _ in range(len(self.get_labels()) + 1)]
        # first column set to frame numbers between first and last frame
        frame_data[0].extend(range(self.get_first_frame(), self.get_last_frame()+1))

        # initiates all column labels to either 0 or 1
        for k in range(len(self.get_labels())):
            frame_data[k + 1].append(self.get_true_false(labels[k], 0))

        # gets the frame start and end of each row in the excel file
        for p in range(len(self.get_column("From"))):
            frame_start.insert(0, self.get_frame_start(p))
            frame_end.insert(0, self.get_frame_end(p))

            # for each frame, puts a 0 or 1 for each column label
            # if you have frames 0 to 10 == 1, 10 to 20 == 0, frame 10 == 1 due to how code is set-up
            for k in range(len(self.get_labels())):
                frame_data[k + 1].extend([self.get_true_false(self.get_labels()[k], p)] * (frame_end[0] - frame_start[0]))

        return frame_data

    def get_first_frame(self):
        # first labeled frame
        first = self.get_column("From").iget(0)
        return first

    def get_last_frame(self):
        # last labeled frame
        last = self.get_column("To").iget(-1)
        return last

    def get_labels(self):
        # column labels of interest for data (ignoring name, lims id, date)
        # make sure to update if columns change
        labels = ["chattering", "trunk_present", "grooming", "trunk_absent", "running",
                  "fidget", "tail_relaxed", "tail_tense", "flailing_present", "flailing_absent", "walking"]
        return labels

    def get_name(self):
        # returns name of annotator
        name = self.get_column("person").iget(0)
        return name

    def get_mouse_ID(self):
        # returns Mouse LIMS ID
        name = self.get_column("mouseid").iget(0)
        return name

    def get_date(self):
        # returns date of annotation
        date= self.get_column("timestamp").iget(0)
        return date


    def store_frame_data(self, label):
        # stores frame data according to whether specified label has 0 or 1 on that frame
        frame_pictures = [[] for _ in range(2)]
        frame_data = self.get_per_frame_data()


        # Iterate through every frame
        for i in range(len(frame_data[0])):
            # i +1 because first frame is at position 1, not 0 in OpenCV
            # read frame at ith position
            self.data_pointer.set(1,i+1)
            ret, frame = self.data_pointer.read()
            # if the label associated with the current frame is equal to zero, place in first column of
            # frame_data, otherwise if its equal to 1 place in second column of frame_data
            # +1 because 0th column on frame_data is frame numbers, label column start at column 1
            if frame_data[self.get_labels().index(label) + 1][i] == 0:
                frame_pictures[0].append(frame)
            elif frame_data[self.get_labels().index(label) +1][i] == 1:
                frame_pictures[1].append(frame)
            else:
                continue

        return frame_pictures

    def is_from_smaller_than_to(self):

        # makes sure the from frame is smaller than the to frame number in a row
        To_frames = self.get_to()
        From_frames = self.get_from()
        # iterates through all rows and does check
        for k in range(len(From_frames)):
            if From_frames[k] == To_frames[k] or From_frames[k] > To_frames[k]:
                return "Frames are timed incorrectly in row %r of the data" %(k+2)
            else:
                continue
        return "All frames are timed correctly!"

    def get_bar_plot(self, label):
        # returns bar plot of frame number and occurrence (0 vs. 1) of a specified column label
        data = self.get_per_frame_data()
        # get frame data
        frames = data[0]
        # get column data
        label_data = data[self.get_labels().index(label)]

        # create plot and axis
        fig1 = plt.figure()
        fig1.suptitle('Occurrence vs. Frame Number', fontsize=14, fontweight='bold')
        ax = fig1.add_subplot(111)
        ax.set_xlabel('frame number')
        ax.set_ylabel('Occurrence')
        ax.bar(frames, label_data)

        return fig1

    def get_cumulative_plot_frame(self, label):
        # returns cumulative sum plot of label occurrence vs. frame number
        data = self.get_per_frame_data()
        frames = data[0]
        # get cumsum of column data
        label_data = np.cumsum(data[self.get_labels().index(label) + 1])
        # create plot axis and figure
        fig1 = plt.figure()
        fig1.suptitle('CumSum of Occurrence vs. Frame Number', fontsize=14, fontweight='bold')
        ax = fig1.add_subplot(111)
        ax.set_xlabel('frame number')
        ax.set_ylabel('CumSum of Occurrence')
        ax.bar(frames, label_data)

        return fig1

    def get_frequency_plot(self, label):
        # returns a plot of the change in frequency of a label per SECOND
        data = self.get_per_frame_data()
        fps = sv(self.directory, self.lims_ID).get_fps()
        # get cumsum of column data
        label_data = np.cumsum(data[self.get_labels().index(label) + 1])
        # initiate counters and lists
        frequency_data = []
        time = []
        count = 0
        n = 0
        # determines over how many frames we calculate annotation frequency
        # 147 frames is approximately 5 seconds
        interval = 147
        interval_seconds = round(147 / fps)

        # iterate over each frame
        for k in range(len(label_data)):
            count += 1
            # if count mod interval size = equal, then calculate the difference of the cumsum associated
            # with this frame minus the cumsum at the frame one interval before
            if count % interval == 0:
                frequency_data.append(label_data[k] - label_data[k - (interval - 1)])
                n += 1
                # round time in seconds to nearest integer
                second = round(n * interval / fps)
                time.append(second)
            else:
                continue

        # create plot axis and figure

        # gets maximum y value (frequency)
        m = max(frequency_data)
        fig1 = plt.figure()
        fig1.suptitle('Frequency of Occurrence, sampled every %r seconds' % (interval_seconds), fontsize=14,
                      fontweight='bold')
        ax = fig1.add_subplot(111)
        ax.set_xlabel('Time (Sec)')
        ax.set_ylabel('Frequency of Occurrence')
        plt.ylim([0, (m + 50)])

        self.create_stimulus_definition(m, ax, fps)

        ax.bar(time, frequency_data)

        return fig1

    def create_stimulus_definition(self, m, ax, fps):
        # create CAM stimulus definition visual

        # get nwb file data
        nwb_file = self.open_nwb()

        #types of possible stimuli
        stimuli = ['spontaneous_stimulus','drifting_gratings_stimulus','natural_movie_one_stimulus', 'natural_movie_two_stimulus',
                   'natural_movie_three_stimulus', 'static_gratings_stimulus',
                   'locally_sparse_noise_stimulus']
        # open data to presentation branch
        visual = nwb_file['stimulus']['presentation']

        # iterate over stimulus types, if type is found in data, then get frame durations
        for stim in stimuli:
            if stim in visual:

                # get unique frame numbers
                frames = np.unique([nwb_file['stimulus']['presentation'][stim]['frame_duration'][()]])

                # to get continuous frame ranges, un-comment code right below
                # ranges = []
                # for k, g in groupby(enumerate(frames), lambda (i, x): i - x):
                #     group = map(itemgetter(1), g)
                #     ranges.append((group[0], group[-1]))

                c = 0
                # diff colors for each stim (black should be for stim that needs an edge color as explained below
                colors = ('y','k','r','b','g','m','c')

                # since some stimuli are on and off frequently (rect too thin), we might want rect edges so that the rect can show
                # Because of edgecolor on
                edgecolor = ('none','k','none','none','none','none','none')


                for i in range(len(frames)-1):
                    # If frame ranges is of length less than 150, then assume it is giving frame ranges. Otherwise,
                    # assume it is giving individual frames ( range e.g. 0 - 10000,
                    # individual e.g. 0, 1, 2, 3.... 10000)
                    # shortest stimulus is 30 seconds, longest movie is 3800 seconds, 3800/30 is about 150
                    # Thus, you wil never have more than 150 discrete frame ranges
                    if len(frames) < 150:
                        length = frames[i+1] - frames[i]
                        x = frames[i] / fps
                        rectangle = plt.Rectangle((x, m + 20), length/fps, 10, fc= colors[stimuli.index(stim)], edgecolor= edgecolor[stimuli.index(stim)],linewidth = 0.5)
                        plt.gca().add_patch(rectangle)

                    else:
                        x = frames[i] / fps
                        rectangle = plt.Rectangle((x, m + 20), 1 / fps, 10, fc=colors[stimuli.index(stim)], edgecolor= edgecolor[stimuli.index(stim)], linewidth = 0.2)
                        plt.gca().add_patch(rectangle)

            # if type if not in data, take next type
            else:
                continue



    def open_nwb(self):
        # opens nwb file
        for file in os.listdir(self.directory):
            if file.endswith("nwb"):
                # make sure file is in there!
                nwb_path = os.path.join(self.directory, file)
                # input file path, r is for read only
                nwb_file = h5py.File(nwb_path, "r")
        if not file:
            print ("nwb file not found.")

        return nwb_file












