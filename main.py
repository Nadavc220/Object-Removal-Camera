"""
About this file:
"""
""" ============================ Imports ============================ """
import logging
import os
import re
from matplotlib import pyplot as plt

import cv2
import numpy as np
import sys
import my_utils as ut

import Video_Detector as vbp
from yolo import yolo_object_detection as yod

""" ============================ Constants ============================ """
FILES_PATH = "files/"
OUTPUT_PATH = "output/"
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
    # image1 = cv2.imread("output/nyc2/50.0.png")
    # image1 = cv2.imread("10.png")
    # detection = yolo_image_detector.detect_frame(image1, ['person'], 0, 0.3)
    # ut.show_image(image1, detections=detection)
    # video_detector.stream_detection("files/ambulance.mov", ['person'])


    for file in files:
        # image1 = cv2.imread("med_result2_0.87.png")
        # image2 = cv2.imread("med_result2_0.5.png")
        # # detections1 = yolo_image_detector.detect_frame(image1, ['person'], detection_resize_factor=0.15)
        # detections2 = yolo_image_detector.detect_frame(image2, ['person'], detection_resize_factor=0.15)
        # # ut.show_image("Asdf", image2, detections=detections2)
        # for detection_list in detections2.unwanted_detections.values():
        #     for detection in detection_list:
        #         box = detection.bounding_box
        #         start_row, end_row, start_col, end_col = box.start_row, box.end_row, box.start_col, box.end_col
        #         patch = image2[start_row:end_row, start_col:end_col]
        #         # cv2.imwrite("patch.png", patch)
        #         background_color = (patch[0][0] + patch[0][-1] + patch[-1][0] + patch[-1, -1]) // 4
        #         background_color_image = np.full((500, 500, 3), background_color)
        #         distance_from_background = np.zeros(patch.shape[:2])
        #         for i in range(patch.shape[0]):
        #             for j in range(patch.shape[1]):
        #                 distance_from_background[i, j] = ut.euclidean_dist(background_color.astype('int'), patch[i, j].astype('int'))
        #         distance_from_background = (distance_from_background * 255 / np.max(distance_from_background)).astype('uint8')
        #         thresh = np.mean(distance_from_background)
        #         print(thresh)
        #         patch[distance_from_background > thresh] = [0, 0, 0]
        #         patch[distance_from_background <= thresh] = [255, 255, 255]
        #         image2[start_row:end_row, start_col:end_col] = patch
        # ut.show_image("Asdf", image2)

        dot_index = re.search("\.", file).start()
        name = file[:dot_index]
        video_detector.file_name = name

        # for alpha in np.arange(init_alpha, final_alpha, 0.01):
        alpha = 0.99
        # initialize directories
        initialize_dirs(alpha, name)
        file_output_folder = OUTPUT_PATH + name + '/'
        # output_path = file_output_folder +
        os.makedirs(os.path.dirname(file_output_folder), exist_ok=True)

        # merge video to image
        video_detector.refresh(file_output_folder, alpha)
        video_detector.object_sensitive_video_merge(FILES_PATH + file, UNWANTED_OBJECTS)
