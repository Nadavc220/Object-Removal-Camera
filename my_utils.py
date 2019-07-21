"""
File: my_utils.py
Description: Holds helpful dara and functions in use of all classes.
Author: Nadav Cohen, nadavc220, 200961969
"""

""" ============================= Imports ============================= """
import numpy as np
import cv2
from scipy.ndimage import imread

""" ============================= Constants ============================= """
GRAY = 0
NO_KEY = 255
BOARDER_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SIZE = 0.6
RED_COLOR = (0, 0, 255)

""" ============================= Classes ============================= """


class DetectionStructure:
    """

    """
    def __init__(self, unwanted_objects, min_confidence=0.55):
        self.detections = {}
        self.unwanted_objects = unwanted_objects
        self.min_conifidence = min_confidence

    def add_detection(self, detection, add_unwanted_only=False):
        if add_unwanted_only and detection.my_class not in self.unwanted_objects:
            # if decided add only unwanted object detections
            return
        if detection.my_class in self.detections:
            self.detections[detection.my_class].append(detection)
        else:
            self.detections[detection.my_class] = [detection]

    def draw_detections_on_image(self, image):
        for class_name in self.detections.keys():
            for detection in self.detections[class_name]:
                detection.draw_on_image(image)

    def update_detections_to_image_coordinates(self, image):
        for my_class in self.detections:
            for detection in self.detections[my_class]:
                detection.bounding_box.update_image_coordinates(image)

    def get_detections(self):
        detection_list = []
        for my_class in self.detections:
            detection_list += self.detections[my_class]
        return detection_list

    def is_pixel_inside_detections(self, row, col):
        for my_class in self.detections:
            for detection in self.detections[my_class]:
                if detection.bounding_box.is_pixel_inside(row, col):
                    return True
        return False

    def get_detected_boxes(self):
        return [detection.bounding_box for my_class in self.detections.keys() for detection in self.detections[my_class]]

    def is_empty(self):
        return len(self.detections) == 0


class Detection:
    """
    A class which represents an object detection
    """
    def __init__(self, my_class, confidence, bounding_box):
        self.my_class = my_class
        self.confidence = confidence
        self.bounding_box = bounding_box

    def draw_on_image(self, image):
        self.bounding_box.update_image_coordinates(image)
        self.bounding_box.draw_on_image(image)
        self.bounding_box.add_text(image, self.my_class)


class Box:
    """
    A class which represents a detection bounding box.
    """
    def __init__(self, start_row, start_col, end_row, end_col, color=RED_COLOR):
        """
        :param start_row:
        :param start_col:
        :param end_row:
        :param end_col:
        :param color: the bounding box color
        """
        self.start_row, self.start_col, self.end_row, self.end_col = start_row, start_col, end_row, end_col
        self.box_color = color
        self.thickness = BOARDER_THICKNESS
        self.text_thickness = TEXT_THICKNESS
        self.text_size = TEXT_SIZE

    def expand(self, ratio):
        """
        Add a ratio of the image size.
        :param ratio: the given ratio (0.1 will add 10% of the image)
        :param limits: the image width and height
        """
        new_start_row = max(self.start_row - (self.end_row - self.start_row) * ratio, 0)
        new_end_row = min(self.end_row + (self.end_row - self.start_row) * ratio, 1)
        new_start_col = max(self.start_col - (self.end_col - self.start_col) * ratio, 0)
        new_end_col = min(self.end_col + (self.end_col - self.start_col) * ratio, 1)
        self.start_row, self.start_col, self.end_row, self.end_col = new_start_row, new_start_col,\
                                                                     new_end_row, new_end_col

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "StartX: " + str(self.start_row) + "; StartY: " + str(self.start_col) + "\n" +\
               "EndX: " + str(self.end_row) + "; EndY: " + str(self.end_col) + "\n"

    def draw_on_image(self, image, full=False):
        if full:
            cv2.rectangle(image, (self.start_col, self.start_row), (self.end_col, self.end_row),
                          (0, 0, 255), -1)
        else:
            cv2.rectangle(image, (self.start_col, self.start_row), (self.end_col, self.end_row),
                          color=self.box_color, thickness=self.thickness)

    def add_text(self, image, text):
        y = self.start_col - 15 if self.start_col - 15 > 0 else self.start_col + 15
        cv2.putText(image, text, (y, self.start_row - 10), cv2.FONT_HERSHEY_SIMPLEX, self.text_size,
                    self.box_color, self.text_thickness)

    def is_pixel_inside(self, x, y):
        return self.start_row <= x <= self.end_row and self.start_col <= y <= self.end_col

    def update_image_coordinates(self, image):
        """
        converts floating values of box to image pixel coordinates.
        :param: image
        """
        max_row, max_col = image.shape[:2]
        self.start_col = int(np.floor(self.start_col * max_col))
        self.start_row = int(self.start_row * max_row)
        self.end_col = min(int(np.ceil(self.end_col * max_col)), max_col - 1)
        self.end_row = min(int(np.ceil(self.end_row * max_row)), max_row - 1)


""" ============================= Functions ============================= """


def draw_detections_on_image(image, detections):
    for class_name in detections.keys():
        for detection in detections[class_name]:
            detection.draw_on_image(image)


def draw_transparent_boxes(image, boxes, alpha=0.2):
    overlay = image.copy()
    output = image.copy()
    for box in boxes:
        box.draw_on_image(overlay, full=True)
        # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def show_image(winname, image, size_factor=1, detections=None, video=False):
    image_copy = np.copy(image)

    # drawing detections on image of given
    if detections is not None:
        detections.draw_detections_on_image(image_copy)

    # resize image with given factor
    shape = (int(image_copy.shape[0] * size_factor), int(image_copy.shape[1] * size_factor))
    resized_image = image_resize(image_copy, shape[0], shape[1])

    # who image and close it instantly if we are streaming a video, else wait for key
    cv2.imshow(winname, resized_image)
    if not video:
        cv2.waitKey(0)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
