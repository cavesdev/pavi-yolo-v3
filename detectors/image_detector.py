"""
Object detection in images with YOLOv3 algorithm.
Class ImageDetector reads an image and passes it to YOLOFrameDetector for processing.

Author: Carlos Cuevas
@cavesdev
February 2020
"""

from .detector import YOLOFrameDetector
import cv2 as cv
import numpy as np
import os
import sys


class ImageDetector:
    def __init__(self):
        self.__detector = YOLOFrameDetector()
        self.__cap = None
        self.__output_file = None

    def load_file(self, filename):
        if not os.path.isfile(filename):
            print("Input image file ", filename, " doesn't exist")
            sys.exit(1)
        self.__cap = cv.VideoCapture(filename)
        self.__output_file = filename[:-4] + '_yolo_out_py.jpg'

    def process(self):
        _, frame = self.__cap.read()

        self.__detector.process(frame)

        cv.imwrite(self.__output_file, frame.astype(np.uint8))
        window_name = 'Deep learning object detection in OpenCV'
        cv.imshow(window_name, frame)
        print("Done processing !!!")
        print("Output file is stored as ", self.__output_file)
        cv.waitKey(5000)

