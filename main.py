"""
About this file:
"""
""" ============================ Imports ============================ """
import cv2
from yolo import yolo_object_detection as yod
import Video_Detector as vbp
import numpy as np
import re
import os
import sys
import logging

""" ============================ Constants ============================ """
FILES_PATH = "files/"
OUTPUT_PATH = "output3/"
FILES = ["train.mov", "nyc2.mp4", "wimbeldon2.mp4"]
# FILES = ["wimbeldon2.mp4"]
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
    if len(sys.argv) <= 1:
        files = FILES
        init_alpha = 0
        final_alpha = 1.01
    else:
        files = [sys.argv[1]]
        init_alpha = float(sys.argv[2])
        final_alpha = float(sys.argv[3])
    yolo_image_detector = yod.FrameDetector()

    for file in files:
        dot_index = re.search("\.", file).start()
        name = file[:dot_index]
        video_detector = vbp.VideoDetector(yolo_image_detector)
        for alpha in np.arange(init_alpha, final_alpha, 0.01):
            initialize_dirs(alpha, name)
            # try:
            output_path = OUTPUT_PATH + name + '/' + str((int(alpha * 100))) + "/"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            video_detector.refresh(output_path, alpha)
            video_detector.get_clean_image(FILES_PATH + file, UNWANTED_OBJECTS)
            # except Exception as e:
            #     logging.info("Exception occurred", exc_info=True)
            #     break
