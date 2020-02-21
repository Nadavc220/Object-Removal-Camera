"""
File: my_utils.py
Description: Holds helpful dara and functions in use of all classes.
Author: Nadav Cohen, nadavc220, 200961969
"""

""" ============================= Imports ============================= """
import numpy as np
import cv2
import math
import logging
from copy import deepcopy


""" ============================= Constants ============================= """
GRAY = 0
NO_KEY = 255
BOARDER_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SIZE = 0.6
RED_COLOR = (0, 0, 1)
PIXEL_SIMILIAR_THRESHOLD = 0.3

""" ============================= Classes ============================= """


class DetectionStructure:
    """

    """
    def __init__(self, unwanted_objects, min_confidence=0.55):
        self.unwanted_detections = {}
        self.wanted_detections = {}
        self.unwanted_objects = unwanted_objects
        self.min_confidence = min_confidence

    def add_detection(self, detection, resize_factor=None):
        if detection.my_class not in self.unwanted_objects:
            if detection.my_class in self.wanted_detections:
                self.wanted_detections[detection.my_class].append(detection)
            else:
                self.wanted_detections[detection.my_class] = [detection]
        else:
            if resize_factor is not None:
                detection.bounding_box.expand(resize_factor)
            if detection.my_class in self.unwanted_detections:
                self.unwanted_detections[detection.my_class].append(detection)
            else:
                self.unwanted_detections[detection.my_class] = [detection]

    def draw_detections_on_image(self, image, unwanted=True, wanted=True):
        if unwanted:
            for class_name in self.unwanted_detections.keys():
                for detection in self.unwanted_detections[class_name]:
                    detection.draw_on_image(image)
        if wanted:
            for class_name in self.wanted_detections.keys():
                for detection in self.wanted_detections[class_name]:
                    detection.draw_on_image(image)

    def update_detections_to_image_coordinates(self, image):
        for my_class in self.unwanted_detections:
            for detection in self.unwanted_detections[my_class]:
                detection.bounding_box.update_image_coordinates(image)
        for my_class in self.wanted_detections:
            for detection in self.wanted_detections[my_class]:
                detection.bounding_box.update_image_coordinates(image)

    def get_detections(self):
        detection_list = []
        for my_class in self.unwanted_detections:
            detection_list += self.unwanted_detections[my_class]
        return detection_list

    def is_pixel_inside_detections(self, row, col):
        for my_class in self.unwanted_detections:
            for detection in self.unwanted_detections[my_class]:
                if detection.bounding_box.is_pixel_inside(row, col):
                    return True
        return False

    def get_detection_map(self, shape, unify_collisions=False):
        shape = shape[0:2]
        detection_map = np.zeros(shape)
        for my_class in self.unwanted_detections.keys():
            for detection in self.unwanted_detections[my_class]:
                for i in range(detection.bounding_box.start_row, detection.bounding_box.end_row + 1):
                    for j in range(detection.bounding_box.start_col, detection.bounding_box.end_col + 1):
                        detection_map[i, j] = 1
                if unify_collisions:
                    for wanted_class in self.wanted_detections.keys():
                        for wanted_detection in self.wanted_detections[wanted_class]:
                            if wanted_detection.bounding_box.is_colliding(detection.bounding_box):
                                for x in range(wanted_detection.bounding_box.start_row,
                                               wanted_detection.bounding_box.end_row + 1):
                                    for y in range(wanted_detection.bounding_box.start_col,
                                                   wanted_detection.bounding_box.end_col + 1):
                                        detection_map[x, y] = 1
        return detection_map

    def get_detected_boxes(self):
        return [detection.bounding_box for my_class in self.unwanted_detections.keys() for detection in self.unwanted_detections[my_class]]

    def is_empty(self):
        return len(self.unwanted_detections) == 0


class Detection:
    """
    A class which represents an object detection
    """
    def __init__(self, my_class, confidence, bounding_box):
        self.my_class = my_class
        self.confidence = confidence
        self.bounding_box = bounding_box

    def draw_on_image(self, image):
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

    def expand_int(self, ratio, shape):
        """
        Add a ratio of the image size.
        :param ratio: the given ratio (0.1 will add 10% of the image)
        :param limits: the image width and height
        """
        new_start_row = int(max(self.start_row - (self.end_row - self.start_row) * ratio, 0))
        new_end_row = int(min(self.end_row + (self.end_row - self.start_row) * ratio, shape[1] - 1))
        new_start_col = int(max(self.start_col - (self.end_col - self.start_col) * ratio, 0))
        new_end_col = int(min(self.end_col + (self.end_col - self.start_col) * ratio, shape[1] - 1))
        self.start_row, self.start_col, self.end_row, self.end_col = new_start_row, new_start_col, \
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

    def check_overlap_percentage(self, other):
        if self.end_col < other.start_col or self.start_col > other.end_col or \
           self.end_row < other.start_row or self.start_row > other.end_row:
            return 0
        # calc common cols
        if self.start_col >= other.start_col and self.end_col <= other.end_col:
            common_cols = self.end_col - self.start_col + 1
        elif self.start_col <= other.start_col and self.end_col >= other.end_col:
            common_cols = other.end_col - other.start_col + 1
        elif self.start_col >= other.start_col:
            common_cols = other.end_col - self.start_col + 1
        else:
            common_cols = self.end_col - other.start_col + 1

        # calc common rows
        if self.start_row >= other.start_row and self.end_row <= other.end_row:
            common_rows = self.end_row - self.start_row + 1
        elif self.start_row <= other.start_row and self.end_row >= other.end_row:
            common_rows = other.end_row - other.start_row + 1
        elif self.start_row >= other.start_row:
            common_rows = other.end_row - self.start_row + 1
        else:
            common_rows = self.end_row - other.start_row + 1

        return ((common_rows * common_cols) / ((self.end_row - self.start_row + 1) * (self.end_col - self.start_col + 1))) * 100

    def is_colliding(self, other):
        return self.check_overlap_percentage(other) > 0


""" ============================= Detection Public Functions ============================= """


def draw_detections_on_image(image, detections):
    for class_name in detections.keys():
        for detection in detections[class_name]:
            detection.draw_on_image(image)


def draw_transparent_pixels(image, pixels, alpha=0.2):
    overlay = image.copy()
    output = image.copy()
    for pixel in pixels:
        overlay[pixel[0], pixel[1]] = RED_COLOR
        # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def draw_transparent_boxes(image, boxes, alpha=0.2):
    overlay = image.copy()
    output = image.copy()
    for box in boxes:
        box.draw_on_image(overlay, full=True)
        # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def unite_detecions(detections):
    """
    Recieves a list of detections and returns a single united detection
    :param detections:
    :return:
    """
    min_confidence = 1
    unwanted_objects = []
    class_detections = {}
    for detection_struct in detections:
        min_confidence = min(min_confidence, detection_struct.min_confidence)
        unwanted_objects += detection_struct.unwanted_objects
        if class_detections == {}:
            class_detections = detection_struct.detections
        else:
            for my_class in detection_struct.detections:
                if my_class not in class_detections.keys():
                    class_detections[my_class] = detection_struct[my_class]
                else:
                    class_detections[my_class] += detection_struct.detections[my_class]
    new = DetectionStructure(list(set(unwanted_objects)), min_confidence)
    new.unwanted_detections = class_detections
    return new


""" ============================= Image Public Functions ============================= """


def show_image(image, size_factor=1, detections=None, video=False):
    image_copy = np.copy(image)

    # drawing detections on image of given
    if detections is not None:
        detections.draw_detections_on_image(image_copy)

    # resize image with given factor
    shape = (int(image_copy.shape[0] * size_factor), int(image_copy.shape[1] * size_factor))
    resized_image = image_resize(image_copy, shape[0], shape[1])

    # show image and close it instantly if we are streaming a video, else wait for key
    cv2.imshow("image", resized_image)
    if not video:
        cv2.waitKey(0)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
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


def get_image_median(frames, pixel_background_frames_map, percentile=0.5):
    gray_frames = [cv2.cvtColor((frame * 255).astype('uint8'), cv2.COLOR_BGR2GRAY) for frame in frames]
    med_image = np.zeros_like(frames[0])
    for row, col in row_col_generator(gray_frames[0].shape, print_percentage=False):
        if (len(pixel_background_frames_map[(row, col)])) == 0:
            pixel_background_frames_map[(row, col)] = list(range(len(frames)))
        intensity_sorted_value = sorted(pixel_background_frames_map[(row, col)],
                                        key=lambda f: gray_frames[f][row, col])
        if len(intensity_sorted_value) > 0:
            frame_index = intensity_sorted_value[int(len(intensity_sorted_value) * percentile)]
            med_image[row, col] = frames[frame_index][row, col]
    return med_image


def get_image_average(frames, pixel_background_frames_map, percentile=0.5):
    med_image = np.zeros_like(frames[0])
    for row, col in row_col_generator(frames[0].shape[:2], print_percentage=False):
        if (len(pixel_background_frames_map[(row, col)])) == 0:
            pixel_background_frames_map[(row, col)] = list(range(len(frames)))
        color_values = [frames[i][row, col] for i in pixel_background_frames_map[(row, col)]]
        med_image[row, col] = np.sum(color_values, axis=0) / len(color_values)
    return med_image * 255


def output_color_distance_maps(self):
    image = self.frames[0]
    for i in np.arange(0.1, 0.5, 0.1):
        copy_im = deepcopy(image)
        for pixel_data in self.pixel_data_dict.values():
            if pixel_data.color_max_distance > i:
                copy_im[pixel_data.row, pixel_data.col] = [0, 255, 0]
        cv2.imwrite(self.output_name_folder + "distance_" + str(int(i * 100)) + ".png",
                    (copy_im * 255).astype('uint8'))
    for i in np.arange(0.5, 1.5, 0.02):
        copy_im = deepcopy(image)
        for pixel_data in self.pixel_data_dict.values():
            if pixel_data.color_max_distance > i:
                copy_im[pixel_data.row, pixel_data.col] = [0, 255, 0]
        cv2.imwrite(self.output_name_folder + "distance_" + str(int(i * 100)) + ".png",
                    (copy_im * 255).astype('uint8'))


def get_pixel_color_dist_map(pixel_data_dict, dist):
    pixels = []
    for pixel_data in pixel_data_dict.values():
        if pixel_data.color_max_distance < dist:
            pixels.append((pixel_data.row, pixel_data.col))
    return pixels


def sort_closest_frames(image, frames, pixel_data_dict, pixel_background_frames_map, color_dist_thresh=0.5):
    pixels_on_check = get_pixel_color_dist_map(pixel_data_dict, color_dist_thresh)
    frame_score_list = []
    for frame_idx in range(len(frames)):
        score = 0
        for pixel in pixels_on_check:
            row, col = pixel
            if frame_idx in pixel_background_frames_map[pixel]:
                score += euclidean_dist(image[row, col], frames[frame_idx][row, col])
        frame_score_list.append((frame_idx, score))
    return [f[0] for f in sorted(frame_score_list, key=lambda x: x[1])]


""" ============================= Pixel Calculations ============================= """


def get_neighbour_indices(row, col, shape):
    neighbour_indices = []
    if row - 1 >= 0:
        neighbour_indices.append((row - 1, col))
    if col - 1 >= 0:
        neighbour_indices.append((row, col - 1))
    if col + 1 < shape[1]:
        neighbour_indices.append((row, col + 1))
    if row + 1 < shape[0]:
        neighbour_indices.append((row + 1, col))
    return neighbour_indices


""" ============================= Log Functions ============================= """


def log_or_print(message, log_msg=False, print_msg=True):
    if log_msg:
        logging.info(message)
    if print_msg:
        print(message)


def log_or_print_time(start, end, log_msg=False, print_msg=True):
    total_secs = end - start
    if total_secs < 60:
        msg = "Done after {:.6f} seconds".format(total_secs)
    elif total_secs < 3600:
        min = total_secs // 60
        secs = total_secs - min * 60
        msg = "Done after " + str(min) + " minutes and " + str(secs) + " seconds"
    else:
        hours = total_secs // 3600
        min = (total_secs - hours * 3600) // 60
        secs = total_secs - hours * 3600 - min * 60
        msg = "Done after " + str(hours) + " hours, " + str(min) + " minutes and " + str(secs) + " seconds"
    log_or_print(msg, log_msg, print_msg)


""" ============================= Pixel Scaling Functions ============================= """


def euclidean_dist(pixel1, pixel2):  # todo temporarily change to manhattan distance
    return math.sqrt((pixel1[0] - pixel2[0])**2 + (pixel1[1] - pixel2[1])**2 + (pixel1[2] - pixel2[2])**2)
    # return np.abs(pixel1[0] - pixel2[0]) + np.abs(pixel1[1] - pixel2[1]) + np.abs(pixel1[2] - pixel2[2])


def euclidean_norm(pixel):
    return math.sqrt(pixel[0]**2 + pixel[1]**2 + pixel[2]**2)


""" ============================= Pixel Iterators ============================= """


def frames_generator(start_row, end_row, start_col, end_col, print_percentage=True):
    count = 0
    while start_row <= end_row and start_col <= end_col:
        for col in range(start_col, end_col + 1):
            yield start_row, col
            count += 1
        for row in range(start_row + 1, end_row + 1):
            yield row, end_col
            count += 1
        for col in reversed(range(start_col - 1, end_col)):
            yield end_row, col
            count += 1
        for row in reversed(range(start_row + 1, end_row)):
            yield row, start_col
            count += 1
        start_row += 1
        end_row -= 1
        start_col += 1
        end_col -= 1
        # if print_percentage:
        #     print("Finished: " + str((count / ((end_row - start_row) * (end_col - start_col) * 100))) + "%")

def row_col_generator(shape, print_percentage=True):
    count = 0
    prev_percentage = 0
    for row in range(shape[0]):
        for col in range(shape[1]):
            yield row, col
            count += 1
            curr_percentage = (count / (shape[0] * shape[1])) * 100
            if print_percentage and int(curr_percentage) > prev_percentage:
                log_or_print("Finished: {:.1f}%".format(curr_percentage, end="", flush=True))
                prev_percentage = curr_percentage
