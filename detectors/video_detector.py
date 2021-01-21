"""
Object detection in video with YOLOv3 algorithm.
Class VideoDetector reads video files and passes frames to YOLOFrameDetector for processing.
If requested, writes a JSON object to a file with:
 - Video metadata
 - Algorithm used
 - Detected classes and number of detections per frame depending on frame-skip configuration.

Author: Carlos Cuevas
@cavesdev
February 2020
"""

from .detector import YOLOFrameDetector
from .config import Config

import cv2 as cv
import numpy as np
import os
import sys
import json
from datetime import datetime


class VideoDetector:
    """Class to process a video by passing its frames to a YOLOFrameDetector"""

    def __init__(self, config_file=None, config_dict=None):
        """
        Set the necessary variables.
        :param config_file: optional configuration file
        """
        if config_file is not None:
            Config.load_from_file(config_file)
        elif config_dict is not None:
            Config.load_from_dict(config_dict)

        self.__frame_detector = YOLOFrameDetector(Config)
        self.__fps = Config.get('fps')
        self.__interval = Config.get('frame-skip')
        self.__cap = None
        self.__vid_writer = None
        self.__video_json = {}
        self.__show = Config.get('show')
        self.__save = Config.get('save')

    def load_file(self, filename):
        """
        Load the video file
        :param filename: filename of the video file to load.
        """
        if not os.path.isfile(filename):
            print("Input video file ", filename, " doesn't exist")
            sys.exit(1)

        self.__cap = cv.VideoCapture(filename)
        output_file = filename[:-4] + Config.get('output-filename') + '.avi'

        if self.__save:
            # Get the video writer initialized to save the output video
            self.__vid_writer = cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.__fps,
                                               (round(self.__cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                                round(self.__cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
        self.__initialize_json(filename)

    def __initialize_json(self, filename):
        """
        Initialize the JSON object with video metadata.
        :param filename: filename of the video file
        :return:
        """
        video_date = 1
        processed_date = datetime.now().__str__()
        filename = os.path.basename(filename)
        name, extension = os.path.splitext(filename)
        tags = None
        duration = cv.CAP_PROP_FRAME_COUNT

        algorithm = dict(
            algorithm='YOLOv3',
            processed_date=processed_date,
            detections=[]
        )

        video_metadata = dict(
            filename=name,
            capture_date=video_date,
            tags=tags,
            duration=duration,
            FPS=self.__fps,
            format=extension,
            processing=[algorithm]
        )

        self.__video_json = video_metadata

    def process(self):
        """
        Main function to process the video file.
        """
        window_name = 'Deep learning object detection in OpenCV'
        if self.__show:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        current_frame = 0

        # main loop
        while cv.waitKey(1) < 0:

            # get frame from the video
            has_frame, frame = self.__cap.read()
            current_frame += 1

            # Stop the program if reached end of video
            if not has_frame:
                print("Done processing !!!")
                cv.waitKey(3000)
                break

            # skip frames
            if current_frame % self.__interval:
                continue

            # process frame
            self.__frame_detector.process(frame)

            # write frame data
            if self.__save:
                self.__vid_writer.write(frame.astype(np.uint8))

            self.__update_json(current_frame)

            if self.__show:
                cv.imshow(window_name, frame)

        self.__cap.release()
        cv.destroyAllWindows()

    def __update_json(self, current_frame):
        """
        Update JSON object with each frame processed
        :param current_frame: processed frame
        """
        frame_json = self.__frame_detector.get_frame_json()
        seconds = current_frame / self.__fps

        data = dict(
            frame=current_frame,
            seconds=seconds,
            objects=frame_json['detections']
        )

        self.__video_json.get('processing')[0].get('detections').append(data)

    def write_json_to_file(self, filename):
        """
        Write JSON object to a file
        :param filename: name of the file to be written
        """
        with open(filename, 'w') as output:
            json.dump(self.__video_json, output, sort_keys=True, indent=2)