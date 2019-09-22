"""
network output info:
detection[0, 0] holds info about all classes:
[1] is the index class (person == 15)
[2] the confidence of this class
[3] - [6] is the bounding box indices (need to round them up)
"""
import imutils

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

    def _filter_boxes(self, boxes, confidences, min_confidence, threshold=0.3):
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)
        idxs = idxs.flatten()
        return idxs

    def _process_network_box_output(self, boxes, confidences, class_ids, unwanted_objects, image_shape, min_confidence, filter=True):
        filtered_boxes_idx = range(len(boxes))
        if filter:
            filtered_boxes_idx = self._filter_boxes(boxes, confidences, min_confidence)
        detection_struct = ut.DetectionStructure(unwanted_objects, min_confidence)
        max_row, max_col = image_shape
        for i in filtered_boxes_idx:
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            start_row = y / max_row
            start_col = x / max_col
            end_row = (y + h) / max_row
            end_col = (x + w) / max_col
            box = ut.Box(start_row, start_col, end_row, end_col, COLORS[class_ids[i]])
            # expanding boxes to make sure entire object is contained in box

            detection = ut.Detection(CLASSES[class_ids[i]], confidences[i], box)
            detection_struct.add_detection(detection)
        return detection_struct

    def detect_frame(self, image, unwanted_objects, min_confidence=0.2):
        """

        :param image:
        :param min_confidence:
        :return:
        """
        start = time.time()

        resized_image = imutils.resize(image, width=400)
        layer_outputs = self._calc_frame_detection(resized_image)
        (max_row, max_col) = resized_image.shape[:2]

        boxes = []
        confidences = []
        class_ids = []
        # check every output layer of the model
        for output in layer_outputs:
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # keeping only good confidence detections
                if confidence > min_confidence:
                    box = detection[0:4] * np.array([max_col, max_row, max_col, max_row])
                    (centerX, centerY, width, height) = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))  # top left corner
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        detection_struct = self._process_network_box_output(boxes, confidences, class_ids, unwanted_objects, resized_image.shape[:2], min_confidence)
        detection_struct.update_detections_to_image_coordinates(image)

        end = time.time()
        print("[INFO] frame processing took {:.6f} seconds".format(end - start))

        return detection_struct

    def detect_multi_frames(self, frames, unwanted_objects, min_confidence=0.2):
        """
        Detects objects of multiple frames and unites the output to single detection
        :param frames:
        :param unwanted_objects:
        :param min_confidence:
        :return:
        """
        detections = []
        for frame in frames:
            detections.append(self.detect_frame(frame, unwanted_objects, min_confidence))
        return ut.unite_detecions(detections)
