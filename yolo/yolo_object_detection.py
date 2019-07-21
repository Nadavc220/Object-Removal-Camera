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
import time

""" ============================ Constants ============================ """

BOARDER_WIDTH = 10
TEXT_WIDTH = 4
TEXT_SIZE = 2
IMAGE_RESIZE_FACTOR = 2
CLASS_INDEX_IDX = 1
CLASS_CONFIDENCE_IDX = 2
BOX_START = 3
BOX_END = 6

WEIGHTS_PATH = "yolo/yolov3.weights"
MODEL_PATH = "yolo/yolov3.cfg"

CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
           "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
           "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
           "toothbrush"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

""" ============================= Classes ============================= """


class FrameDetector:
    """
    A class which represents a detector of single frames
    """
    def __init__(self):
        self.net = self.initialze_network()
        self.text_width = TEXT_WIDTH
        self.text_size = TEXT_SIZE
        self.boarder_width = BOARDER_WIDTH
        self.image_resize_factor = IMAGE_RESIZE_FACTOR

    @staticmethod
    def initialze_network():
        """
        Intializes a Mobilenet trained network and returns it.
        :return: an initialized Mobilenet trained network.
        """
        print("[INFO] loading model... [INFO]")
        net = cv2.dnn.readNetFromDarknet(MODEL_PATH, WEIGHTS_PATH)
        return net

    def _get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def _calc_frame_detection(self, frame, show_info=False):
        """
        construct an input blob for the image
        by resizing to a fixed 300x300 pixels and then normalizing it
        (note: normalization is done via the authors of the MobileNet SSD
        implementation)
        :param frame: the image we want to use for detection.
        :return:
        """
        start = time.time()
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        if show_info:
            print('[INFO] computing object detection... [INFO]')
        self.net.setInput(blob)
        detections = self.net.forward(self._get_output_layers())
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        return detections

    def detect_frame(self, image, unwanted_objects, min_confidence=0.2):
        """

        :param image:
        :param min_confidence:
        :return:
        """
        layer_outputs = self._calc_frame_detection(image)

        start = time.time()
        detection_struct = ut.DetectionStructure(unwanted_objects, min_confidence)
        (max_row, max_col) = image.shape[:2]
        # check every output layer of the model
        for output in layer_outputs:
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > min_confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height, afterwords return
                    # it to raw coordinates so it can be resized to different shapes.
                    box = detection[0:4] * np.array([max_col, max_row, max_col, max_row])
                    (centerX, centerY, width, height) = box.astype("int")
                    start_row = int(centerY - (height / 2)) / max_row
                    start_col = int(centerX - (width / 2)) / max_col
                    end_row = int(centerY + (height / 2)) / max_row
                    end_col = int(centerX + (width / 2)) / max_col
                    box = ut.Box(start_row, start_col, end_row, end_col, COLORS[class_id])
                    # expanding boxes to make sure entire object is contained in box
                    box.expand(0.2)

                    detection = ut.Detection(CLASSES[class_id], confidence, box)
                    detection_struct.add_detection(detection, add_unwanted_only=True)

        end = time.time()
        print("[INFO] frame processing took {:.6f} seconds".format(end - start))
        return detection_struct
