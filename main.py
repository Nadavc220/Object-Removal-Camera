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
OUTPUT_PATH = "output/"
FILES = ["train.mov", "nyc2.mp4", "wimbeldon2.mp4"]
# FILES = ["wimbeldon2.mp4"]
UNWANTED_OBJECTS = ["person"]
""" ============================ Functions ============================ """


def initialize_dirs(alpha, name):
    output_path = OUTPUT_PATH + name + '/' + str(alpha) + "/"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_name = "alpha_" + str(alpha) + ".log"
    logger = logging.getLogger()
    logger.handlers = []
    logging.basicConfig(filename=output_path + log_name, level=logging.INFO)
    logging.info("Staring")


if __name__ == '__main__':
    # Receive arguments
    if len(sys.argv) <= 1:
        files = FILES
        init_alpha = 1000
        final_alpha = 5000
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

        for alpha in np.arange(init_alpha, final_alpha, 100):
            # initialize directories
            initialize_dirs(alpha, name)
            file_output_folder = OUTPUT_PATH + name + '/'
            output_path = file_output_folder + str(alpha) + "/"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # merge video to image
            video_detector.refresh(file_output_folder, alpha)
            video_detector.object_sensitive_video_merge(FILES_PATH + file, UNWANTED_OBJECTS)
