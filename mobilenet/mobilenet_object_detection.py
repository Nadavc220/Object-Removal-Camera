"""
network output info:
detection[0, 0] holds info about all classes:
[1] is the index class (person == 15)
[2] the confidence of this class
[3] - [6] is the bounding box indices (need to round them up)
"""

""" ============================ Imports ============================ """
import numpy as np
import cv2
import my_utils as ut

""" ============================ Constants ============================ """

BOARDER_WIDTH = 10

IMAGE_RESIZE_FACTOR = 2
CLASS_INDEX_IDX = 1
DETECTION_CONFIDENCE_IDX = 2
BOX_START_INDEX = 3
BOX_END_INDEX = 6

PROTOTXT_PATH = "mobilenet/MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "mobilenet/MobileNetSSD_deploy.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

""" ============================= Classes ============================= """


class FrameDetector:
    """
    A class which represents a detector of single frames
    """
    def __init__(self):
        self.net = self.initialze_network()
        self.boarder_width = BOARDER_WIDTH
        self.image_resize_factor = IMAGE_RESIZE_FACTOR

    @staticmethod
    def initialze_network():
        """
        Intializes a Mobilenet trained network and returns it.
        :return: an initialized Mobilenet trained network.
        """
        print("[INFO] loading model... [INFO]")
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        return net

    def __calc_frame_detection(self, frame, show_info=False):
        """
        construct an input blob for the image
        by resizing to a fixed 300x300 pixels and then normalizing it
        (note: normalization is done via the authors of the MobileNet SSD
        implementation)
        :param frame: the image we want to use for detection.
        :return:
        """
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the neural network
        if show_info:
            print('[INFO] computing object detection... [INFO]')
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections

    def detect_frame(self, image, min_confidence=0.2):
        """
        Detects an image for objects.
        :param image: the given image to detect.
        :param min_confidence: the minimal confidence for detecting an object.
        :return: A dictionary which for each detected class holds a list of its Detect objects.
        """
        # (max_row, max_col) = image.shape[:2]
        detections = self.__calc_frame_detection(image)

        # loop over the detections and add confident detections
        detection_dict = {}
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., the probability) associated with the prediction
            confidence = detections[0, 0, i, DETECTION_CONFIDENCE_IDX]

            # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
            if confidence > min_confidence:
                # extract the index of the classes label from the 'detections',
                # The box coordinates are left raw, so it could be scaled to the image in different sizes.
                idx = int(detections[0, 0, i, CLASS_INDEX_IDX])
                start_col, start_row, end_col, end_row = detections[0, 0, i, BOX_START_INDEX:BOX_END_INDEX + 1]
                box = ut.Box(start_row, start_col, end_row, end_col, COLORS[idx])
                box.expand(0.2)  # expanding boxes to make sure entire object is contained in box

                # Add detection to returned info dictionary
                detection = ut.Detection(CLASSES[idx], confidence, box)
                if CLASSES[idx] in detection_dict:
                    detection_dict[CLASSES[idx]].append(detection)
                else:
                    detection_dict[CLASSES[idx]] = [detection]

        return detection_dict
