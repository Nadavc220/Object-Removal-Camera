"""
About this file:
"""
""" ============================ Imports ============================ """
import logging
import os
import re

import cv2
import numpy as np
import sys

import Video_Detector as vbp
from yolo import yolo_object_detection as yod

""" ============================ Constants ============================ """
FILES_PATH = "files/"
OUTPUT_PATH = "output3/"
# FILES = ["train.mov", "nyc2.mp4", "wimbeldon2.mp4"]
FILES = ["nyc2.mp4"]
UNWANTED_OBJECTS = ["person"]
""" ============================ Functions ============================ """


def initialize_dirs(alpha, name):
    alpha_int_value = int(alpha * 100)
    output_path = OUTPUT_PATH + name + '/' + str(alpha_int_value) + "/"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_name = "alpha_" + str(alpha_int_value) + ".log"
    logger = logging.getLogger()
    logger.handlers = []
    logging.basicConfig(filename=output_path + log_name, level=logging.INFO)
    logging.info("Staring")


if __name__ == '__main__':
    # Receive arguments
    if len(sys.argv) <= 1:
        files = FILES
        init_alpha = 0
        final_alpha = 1.01
    else:
        files = [sys.argv[1]]
        init_alpha = float(sys.argv[2])
        final_alpha = float(sys.argv[3])

    # Initialize Object Detectors
    yolo_image_detector = yod.FrameDetector()
    video_detector = vbp.VideoDetector(yolo_image_detector)

    for file in files:
        dot_index = re.search("\.", file).start()
        name = file[:dot_index]

        for alpha in np.arange(init_alpha, final_alpha, 0.01):
            # initialize directories
            initialize_dirs(alpha, name)
            file_output_folder = OUTPUT_PATH + name + '/'
            output_path = file_output_folder + str((int(alpha * 100))) + "/"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # merge video to image
            video_detector.refresh(file_output_folder, alpha)
            video_detector.object_sensitive_video_merge(FILES_PATH + file, UNWANTED_OBJECTS)
