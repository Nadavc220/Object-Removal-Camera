"""
File: my_utils.py
Description: Holds helpful dara and functions in use of all classes.
Author: Nadav Cohen, nadavc220, 200961969
"""

""" ============================= Imports ============================= """
import numpy as np
import cv2
import math
from Pixel_Data import *
import logging

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

    def add_detection(self, detection):
        if detection.my_class not in self.unwanted_objects:
            if detection.my_class in self.wanted_detections:
                self.wanted_detections[detection.my_class].append(detection)
            else:
                self.wanted_detections[detection.my_class] = [detection]
        else:
            detection.bounding_box.expand(-0.1)
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


""" ============================= Functions ============================= """




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


def show_image(winname, image, size_factor=1, detections=None, video=False):
    image_copy = np.copy(image)

    # drawing detections on image of given
    if detections is not None:
        detections.draw_detections_on_image(image_copy)

    # resize image with given factor
    shape = (int(image_copy.shape[0] * size_factor), int(image_copy.shape[1] * size_factor))
    resized_image = image_resize(image_copy, shape[0], shape[1])

    # show image and close it instantly if we are streaming a video, else wait for key
    cv2.imshow(winname, resized_image)
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


""" ============================= Pixel Calculations ============================= """

#
# def calculate_common_pixel(row, col, background_map, frames):
#     accurences = {}
#     mapping = background_map[(row, col)]
#     for frame_idx in mapping:
#         value = frames[frame_idx][row, col]
#         tup = (value[0], value[1], value[2])
#         found = False
#         for accurence in accurences.keys():
#             if pixel_dist(np.array(accurence).astype('int'), np.array(tup).astype('int')) < PIXEL_SIMILIAR_THRESHOLD:
#                 accurences[accurence].append(tup)
#                 found = True
#                 break
#         if not found:
#             accurences[tup] = [tup]
#     best_key = sorted(accurences.keys(), key=lambda x: len(accurences[x]), reverse=True)[0]
#     best_key_pixels = accurences[best_key]
#     # sorted_chosen_pixels = sorted(best_key_pixels, key=lambda x: pixel_norm(x))
#     # return sorted_chosen_pixels[len(sorted_chosen_pixels) // 2]
#     r, g, b = 0, 0, 0
#     for pixel in best_key_pixels:
#         r += pixel[0] / len(best_key_pixels)
#         g += pixel[1] / len(best_key_pixels)
#         b += pixel[2] / len(best_key_pixels)
#     return np.array([int(r), int(g), int(b)])
#
#
# def calculate_pixel_clusters(row, col, background_map, frames, curr_image):
#     background_frame_indices = background_map[(row, col)]  # a list of frames this pixel was in the background
#     # initialize cluster dictionary
#     keys = [(pixel[0], pixel[1], pixel[2]) for pixel in [frames[i][row, col] for i in background_frame_indices]]
#     pixel_clusters = {k: [np.array(k)] for k in keys}
#     for i in range(len(keys)):
#         for j in range(i + 1, len(keys)):
#             pixel, other = keys[i], keys[j]
#             if euclidean_dist(pixel, other) <= PIXEL_SIMILIAR_THRESHOLD:
#                 pixel_clusters[pixel].append(np.array(other))
#                 pixel_clusters[other].append(np.array(pixel))
#     return np.array(cluster_size_pixel_sort(pixel_clusters))


# def calculate_pixel_options(row, col, background_map, frames):
#     """
#     Calculates a list of [(p, cs)]
#     :param row:
#     :param col:
#     :param background_map:
#     :param frames:
#     :return:
#     """
#     pixel_clusters = calculate_pixel_clusters(row, col, background_map, frames)
#     pixel_clusters_sizes = [(k, len(pixel_clusters[k])) for k in
#                             sorted(pixel_clusters.keys(), key=lambda k: len(pixel_clusters[k]), reverse=True)]
#     return pixel_clusters_sizes


def calculate_pixel_clusters(row, col, background_map, frames):
    """
    Calculates a dictionary of {pixel: cluster} here pixel is the frame number the pixel was taken from and
    cluster is all other frames which their pixels are close up to a threshold to the key pixel.
    :param row: row of the pixel
    :param col: col of the pixel
    :param background_map: the mapping of the frames which represent a background value for this pixel
    :param frames: all frames gathered from the video.
    :return: A dictionary of pixels and their clusters
    """
    background_frame_indices = background_map[(row, col)]  # a list of frames this pixel was in the background
    pixel_clusters = {k: [k] for k in background_frame_indices}
    for i in range(len(background_frame_indices)):
        for j in range(i + 1, len(background_frame_indices)):
            frame_index = background_frame_indices[i]
            other_index = background_frame_indices[j]
            if euclidean_dist(frames[frame_index][row, col], frames[other_index][row, col]) <= PIXEL_SIMILIAR_THRESHOLD:
                pixel_clusters[frame_index].append(other_index)
                pixel_clusters[other_index].append(frame_index)
    return process_unique_pixel_clusters(pixel_clusters, row, col, frames)


def process_unique_pixel_clusters(pixel_clusters, row, col, frames):
    filtered_dict = {}
    seen_color_values = []

    for key in pixel_clusters:
        color_tup = tuple(frames[key][row, col])
        if color_tup not in seen_color_values:
            filtered_dict[key] = pixel_clusters[key]
            seen_color_values.append(color_tup)
    return filtered_dict


def calculate_pixel_data(row, col, background_map, frames):
    """
    Creates a PixelData object from video data gathered.
    :param row: row of the pixel
    :param col: col of the pixel
    :param background_map: the mapping of the frames which represent a background value for this pixel
    :param frames: all frames gathered from the video.
    :return: PixelData object of the given row, col pixel
    """
    pixel_clusters = calculate_pixel_clusters(row, col, background_map, frames)
    sorted_frame_cluster_list = [(k, len(pixel_clusters[k])) for k in
                     sorted(pixel_clusters.keys(), key=lambda k: len(pixel_clusters[k]), reverse=True)]
    frame_to_cluster_size_dict = {p[0]: p[1] for p in sorted_frame_cluster_list}
    source_frame = sorted_frame_cluster_list[0][0]
    return PixelData(row, col, source_frame, frame_to_cluster_size_dict, sorted_frame_cluster_list)




        # value = frames[frame_idx][row, col]
        # tup = (value[0], value[1], value[2])
        # found = False
        # for accurence in clusters.keys():
        #     if euclidean_dist(np.array(accurence), np.array(tup)) < PIXEL_SIMILIAR_THRESHOLD:
        #         clusters[accurence].append(value)
        #         found = True
        #         break
        # if not found:
        #     clusters[tup] = [value]
    # if True:
    #     best_key = calculate_best_color_key_commonwise(clusters)
    #     best_key_pixels = clusters[best_key]
    #     # sorted_chosen_pixels = sorted(best_key_pixels, key=lambda x: pixel_norm(x))
    #     # return sorted_chosen_pixels[len(sorted_chosen_pixels) // 2]
    #     r, g, b = 0, 0, 0
    #     for pixel in best_key_pixels:
    #         r += pixel[0] / len(best_key_pixels)
    #         g += pixel[1] / len(best_key_pixels)
    #         b += pixel[2] / len(best_key_pixels)
    #     return np.array([r, g, b])
    # else:
    #     best_key = calculate_best_color_key_neighbourwise(row, col, clusters, curr_image)
    #     return clusters[best_key]
#
# def calculate_common_pixel(row, col, background_map, frames, curr_image):
#     accurences = {}  # clustering pixels ny euclidean distance
#     mapping = background_map[(row, col)]  # a list of frames this pixel was in the background
#     for frame_idx in mapping:
#         value = frames[frame_idx][row, col]
#         tup = (value[0], value[1], value[2])
#         found = False
#         for accurence in accurences.keys():
#             if pixel_dist(np.array(accurence), value) < PIXEL_SIMILIAR_THRESHOLD:
#                 accurences[accurence].append(value)
#                 found = True
#                 break
#         if not found:
#             accurences[tup] = [value]
#         best_key = calc_best_key_spatial(row, col, accurences, mapping, frames)
#         return sorted(accurences[best_key], key=pixel_norm)[len(accurences[best_key]) // 2]
#
#
# def calc_best_key_spatial(row, col, accurences, mapping, frames):
#     spatial_pixels = get_spatial_values(row, col, mapping, frames)
#     count_dict = {}
#     for key in accurences.keys():
#         count_dict[key] = 0
#         for pixel in spatial_pixels:
#             if euclidean_dist(key, pixel) < PIXEL_SIMILIAR_THRESHOLD:
#                 count_dict[key] += 1
#     return sorted(count_dict.keys(), key=lambda x: count_dict[x], reverse=True)[0]


# def cluster_size_pixel_sort(row, col, pixel_clusters, frames):
#     best_key = sorted(pixel_clusters.keys(), key=lambda x: len(pixel_clusters[x]), reverse=True)[0]
#     # close_frame_indices = pixel_clusters[best_key]
#     # average_cluster_value = np.average([frames[i][row, col] for i in close_frame_indices], axis=0)
#     average_cluster_value = frames[best_key][row, col]
#     return average_cluster_value


# def calculate_best_color_key_neighbourwise(row, col, accurences, curr_image):
#     neighbour_values = get_neighbour_values(row, col, curr_image)
#     if len(neighbour_values) == 0:
#         return calculate_best_color_key_commonwise(accurences)
#     for key in accurences:
#         accurences[key] = sum(accurences[key]) // len(accurences[key])
#     sorted_neigh = sorted(neighbour_values, key=lambda x: x[1], reverse=True)
#     for val, count in sorted_neigh:
#         for acc in accurences.keys():
#             if euclidean_dist(val, accurences[acc]) < PIXEL_SIMILIAR_THRESHOLD:
#                 return acc
#     return calculate_best_color_key_commonwise(accurences)


# def get_neighbour_values(row, col, image):
#     final_vals = {}
#     neighbour_vals = []
#     if row - 1 >= 0 and col - 1 >= 0:
#         neighbour_vals.append(image[row - 1, col - 1])
#     if row - 1 >= 0:
#         neighbour_vals.append(image[row - 1, col])
#     if row - 1 >= 0 and col + 1 < image.shape[1]:
#         neighbour_vals.append(image[row - 1, col + 1])
#     if col - 1 >= 0:
#         neighbour_vals.append(image[row, col - 1])
#     if col + 1 < image.shape[1]:
#         neighbour_vals.append(image[row, col + 1])
#     if col - 1 >= 0 and row + 1 < image.shape[0]:
#         neighbour_vals.append(image[row + 1, col - 1])
#     if row + 1 < image.shape[0]:
#         neighbour_vals.append(image[row + 1, col])
#     if row + 1 < image.shape[0] and col + 1 < image.shape[1]:
#         neighbour_vals.append(image[row + 1, col + 1])
#
#     neighbour_vals = list(filter(lambda x: not np.any(x == -1), neighbour_vals))
#     if len(neighbour_vals) <= 1:
#         return [(val, 1) for val in neighbour_vals]
#
#     for pix in neighbour_vals:
#         pix_tup = (pix[0], pix[1], pix[2])
#         for val in final_vals.keys():
#             if euclidean_dist(val, pix_tup) < PIXEL_SIMILIAR_THRESHOLD:
#                 final_vals[val].append(pix_tup)
#                 continue
#         final_vals[pix_tup] = [pix]
#     return [(sum(final_vals[key]) // len(final_vals[key]), len(final_vals[key])) for key in final_vals.keys()]

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



# def get_spatial_values(row, col, mapping, frames):
#     spatial = []
#     shape = frames[0].shape
#     start_row = max(0, row - 5)
#     start_col = max(0, col - 5)
#     end_row = min(shape[0] - 1, row + 5)
#     end_col = min(shape[1] - 1, col + 5)
#     for frame_idx in mapping:
#         frame = frames[frame_idx]
#         patch = frame[start_row:end_row, start_col:end_col]
#         reshaped_patch = patch.reshape((patch.shape[0] * patch.shape[1], 3))
#         for pixel in reshaped_patch:
#             spatial.append(pixel)
#     return spatial


def log_or_print(message, log_msg=True, print_msg=True):
    if log_msg:
        logging.info(message)
    if print_msg:
        print(message)


def log_or_print_time(start, end, log_msg=True, print_msg=True):
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


def euclidean_dist(pixel1, pixel2):
    return math.sqrt((pixel1[0] - pixel2[0])**2 + (pixel1[1] - pixel2[1])**2 + (pixel1[2] - pixel2[2])**2)


def euclidean_norm(pixel):
    return math.sqrt(pixel[0]**2 + pixel[1]**2 + pixel[2]**2)


""" ============================= Pixel Iterators ============================= """


def frames_generator(shape, print_percentage=True):
    count = 0
    row_lower_lim, row_upper_lim, col_lower_lim, col_upper_lim = 0, shape[0], 0, shape[1]
    while row_lower_lim < row_upper_lim:
        for col in range(col_lower_lim, col_upper_lim):
            yield row_lower_lim, col
            count += 1
        for row in range(row_lower_lim + 1, row_upper_lim):
            yield row, col_upper_lim - 1
            count += 1
        for col in reversed(range(col_lower_lim, col_upper_lim - 1)):
            yield row_upper_lim - 1, col
            count += 1
        for row in reversed(range(row_lower_lim + 1, row_upper_lim - 1)):
            yield row, col_lower_lim
            count += 1
        row_lower_lim += 1
        row_upper_lim -= 1
        col_lower_lim += 1
        col_upper_lim -= 1
        if print_percentage:
            print("Finished: " + str((count / (shape[0] * shape[1])) * 100) + "%")


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
