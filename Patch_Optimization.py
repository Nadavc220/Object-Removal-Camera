import my_utils as ut
import sys
import cv2
import os
from Pixel_Data import *
from yolo import yolo_object_detection as yod
import numpy as np

UNWANTED_OBJECTS = ["person"]
BASE_DIRECTORY = 'Patch_Optimization_Output/'


def image_euclid_dist(image1, image2, effective_pixels=None):
    assert image1.shape == image2.shape
    if effective_pixels is None:
        effective_pixels = np.ones(image1.shape)
    if np.max(image1) <= 1:
        image1 = image1 * 255
    if np.max(image2) <= 1:
        image2 = image2 * 255

    sum = 0
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if np.any(effective_pixels[i, j][0] == 1):  # todo - this slows down the process very much!
                sum += ut.euclidean_dist(image1[i][j].astype('int'), image2[i][j].astype('int'))
    return sum


class PatchOptimizer:
    def __init__(self, object_detector, frames, video_name, used_close_patches_percentage):
        self.object_detector = object_detector

        self.frames = frames
        if np.max(self.frames[0]) <= 1:
            for i in range(len(self.frames)):
                self.frames[i] = (self.frames[i] * 255).astype('uint8')

        self.video_name = video_name
        self.used_close_patches_percentage = used_close_patches_percentage
        self.detection_patch_dict = None
        # self.current_percentage_output_dir = None

        self.output_dir = BASE_DIRECTORY + self.video_name + "/"
        os.makedirs(self.output_dir, exist_ok=True)

    def optimize_wanted_object_patches(self, image, pixel_background_frames_map):
        print("=========================================================")
        print("Starting Patch Wanted Object Optimization")
        if np.max(image) <= 1:
            image = (image * 255).astype('uint8')
        image = image.astype('uint8')
        detections = self.object_detector.detect_frame(image, UNWANTED_OBJECTS, min_confidence=0.3)

        copy_image = np.copy(image)
        for my_class in detections.wanted_detections.keys():
            detection_count = 0
            detection_class_list = detections.wanted_detections[my_class]
            for detection in detection_class_list:
                box = detection.bounding_box
                orig_patch = image[box.start_row: box.end_row, box.start_col: box.end_col]
                new_patch = self.__get_patch_optimal_combination(orig_patch, box.start_row, box.end_row, box.start_col, box.end_col, pixel_background_frames_map)
                copy_image[box.start_row: box.end_row, box.start_col: box.end_col] = new_patch
        for my_class in detections.unwanted_detections.keys():
            self.used_close_patches_percentage = 0.05
            detection_count = 0
            detection_class_list = detections.unwanted_detections[my_class]
            for detection in detection_class_list:
                box = detection.bounding_box
                box.expand_int(0.1, image.shape)
                orig_patch = image[box.start_row: box.end_row, box.start_col: box.end_col]
                new_patch = self.__get_patch_optimal_combination(orig_patch, box.start_row, box.end_row, box.start_col, box.end_col, pixel_background_frames_map)
                copy_image[box.start_row: box.end_row, box.start_col: box.end_col] = new_patch
        return copy_image

    def optimize_unwanted_objects(self, image):
        print("=========================================================")
        print("Starting Patch Unwanted Object Optimization")
        if np.max(image) <= 1:
            image = (image * 255).astype('uint8')
        image = image.astype('uint8')
        detections = self.object_detector.detect_frame(image, UNWANTED_OBJECTS, min_confidence=0.3)

        copy_image = np.copy(image)
        for my_class in detections.unwanted_detections.keys():
            detection_count = 0
            detection_class_list = detections.unwanted_detections[my_class]
            for detection in detection_class_list:
                box = detection.bounding_box
                copy_image = self.__complete_patch_framewise(box.start_row, box.end_row, box.start_col, box.end_col, copy_image)
        return copy_image

    def optimize_background_patches(self, image):
        print("=========================================================")
        print("Starting Patch background optimization...")
        if np.max(image) <= 1:
            image = (image * 255).astype('uint8')

        # patch_sequence = [120, 100, 90, 60, 40, 20, 10]
        patch_sequence = [40]
        # we mark all pixels we don't want to change in this iterations: objects and pure background pixels
        image = image.astype('uint8')
        detections = self.object_detector.detect_frame(image, [], min_confidence=0.3)
        effective_pixels = np.ones(image.shape)
        # mark all object pixels
        for my_class in detections.wanted_detections.keys():
            for detection in detections.wanted_detections[my_class]:
                box = detection.bounding_box
                effective_pixels[box.start_row: box.end_row + 1, box.start_col: box.end_col + 1] = 0

        # save the changed pixels values dor later
        image2 = np.copy(image)
        image2[effective_pixels == 0] = 0
        cv2.imwrite(self.output_dir + "changed_pixels.png", image2)

        # process with multiple patch sizes
        for patch_size in patch_sequence:
            print("=========================================================")
            print("running Patch size " + str(patch_size))
            for start_row in range(0, image.shape[0], patch_size):
                print(str((start_row / image.shape[0]) * 100) + "%")
                end_row = min(start_row + patch_size, image.shape[0])
                for start_col in range(0, image.shape[1], patch_size):
                    end_col = min(start_col + patch_size, image.shape[1])

                    # extract original patch for finding closest patches
                    orig_patch = image[start_row: end_row, start_col: end_col]

                    # retrive the map of effective pixels for the patch
                    effective_patch_pixels = effective_pixels[start_row: end_row, start_col: end_col].astype('uint8')

                    # calculate new patch from all frames
                    new_patch = self.__get_patch_optimal_combination(orig_patch, start_row, end_row, start_col, end_col, effective_patch_pixels)  # todo: this is in float, need to change frames to int, edit: not anymore

                    # calculate the negated map of not effective pixels
                    non_effective_pixels = np.ones(effective_patch_pixels.shape).astype('uint8') - effective_patch_pixels

                    # turn changed values to 0 (by multiplying with 0) and keep uneffective patches the same (multi by 1)
                    image[start_row: end_row, start_col: end_col] = orig_patch * non_effective_pixels
                    # turn to zero all pixels we dont want to change
                    new_patch = new_patch * effective_patch_pixels
                    # add the new patch to the image (non effective pixels are added zero and stay unchanged from before)
                    image[start_row: end_row, start_col: end_col] += new_patch.astype('uint8')
            cv2.imwrite(BASE_DIRECTORY + self.video_name + "/" + str(patch_size) + ".png", image)

    def __get_patch_optimal_combination(self, orig_patch, start_row, end_row, start_col, end_col, pixel_background_frames_map, effective_patch_pixels=None):
        sorted_similiar_patches = self.__get_distance_sorted_patches(orig_patch, start_row, end_row, start_col, end_col, effective_patch_pixels)
        # use only a certain percentage of the given patches
        effective_similiar_patches = sorted_similiar_patches[:int(len(sorted_similiar_patches) * self.used_close_patches_percentage)]
        # return their weighted average result
        patches = [patch_info[0].astype('uint8') for patch_info in effective_similiar_patches]
        scores = [patch_info[1] for patch_info in effective_similiar_patches]
        if sum(scores) == 0:
            return np.average(patches, axis=0).astype('uint8')
        # score_sum = sum([patch_info[1] for patch_info in effective_similiar_patches])
        # patch_weights = [patch_info[1] / score_sum for patch_info in effective_similiar_patches]
        return np.average(patches, axis=0, weights=scores).astype('uint8')  # remove weights for normal average
        # return ut.get_image_median(patches, pixel_background_frames_map=pixel_background_frames_map).astype('uint8')  # remove weights for normal average

    def __get_distance_sorted_patches(self, orig_patch, start_row, end_row, start_col, end_col, effective_patch_pixels):
        patches = []
        for frame in self.frames:
            curr_patch = frame[start_row: end_row, start_col: end_col]
            score = image_euclid_dist(orig_patch, curr_patch, effective_pixels=effective_patch_pixels)
            patches.append((curr_patch, score))
        return sorted(patches, key=lambda x: x[1])

    def __complete_patch_framewise(self, start_row, end_row, start_col, end_col, image):
        image_mask = np.ones(image.shape[:2])
        image_mask[start_row: end_row + 1, start_col: end_col + 1] = 0
        best_frames_dict = {}
        for pixel in ut.frames_generator(start_row, end_row, start_col, end_col, print_percentage=False):
            neighbors = ut.get_neighbour_indices(pixel[0], pixel[1], image.shape)
            valid_neighbors = []
            for neighbor in neighbors:
                if image_mask[neighbor[0], neighbor[1]] == 1:
                    valid_neighbors.append(neighbor)
            best_sum = -1
            best_frame = 0
            for f in range(len(self.frames)):
                frame = self.frames[f]
                curr_sum = 0
                for neighbor in valid_neighbors:
                    curr_sum += ut.euclidean_dist(frame[pixel[0], pixel[1]].astype('int'), image[neighbor[0], neighbor[1]].astype('int'))
                if curr_sum < best_sum or best_sum == -1:
                    best_sum = curr_sum
                    best_frame = f
            image_mask[pixel[0], pixel[1]] = 1
            best_frames_dict[pixel] = best_frame
        for pixel in best_frames_dict.keys():
            image[pixel[0], pixel[1]] = self.frames[best_frames_dict[pixel]][pixel[0], pixel[1]]
        return image

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ OLD CODE

    # def get_video_frames(video_path):
    #     fdp = 20
    #     print("getting frames...")
    #     frames = []
    #     vs = cv2.VideoCapture(video_path)
    #     ret, frame = vs.read()
    #
    #     count = 0
    #     while ret:
    #         if count % fdp == 0:
    #             frames.append(frame)
    #     print("done")
    #     return frames

    # class PatchAlternative:
    #
    #     def __init__(self, patch, score):
    #         self.patch = patch
    #         self.score = score
    #
    #
    # class PatchAlternativesCollection:
    #
    #     def __init__(self, original_patch, alternatives, start_row, start_col):
    #         self.original_patch = original_patch
    #         self.alternatives = alternatives
    #         self.start_row = start_row
    #         self.start_col = start_col


    # def optimize_object_patches(self, image, save_similiar_patches=False):
    #     print("=========================================================")
    #     print("Starting Patch Object Optimization")
    #     if np.max(image) <= 1:
    #         image = (image * 255).astype('uint8')
    #     detections = self.object_detector.detect_frame(image, UNWANTED_OBJECTS, min_confidence=0.3)
    #     if self.detection_patch_dict is None:  # todo: check if you can use this in next function
    #         self.__find_similar_patches(image, detections, save_patches=save_similiar_patches)
    #     # self.current_percentage_output_dir = BASE_DIRECTORY + str(self.used_close_patches_percentage * 100) + "/"
    #     # os.makedirs(self.current_percentage_output_dir, exist_ok=True)  #todo: erase line
    #     return self.__optimize_patches(image)
    #
    #
    #
    # def __find_similar_patches(self, image, detections, save_patches=False):
    #     self.detection_patch_dict = {}
    #     output_dir = BASE_DIRECTORY + 'similar_patches/'
    #     print("Starting to find patches for " + str(len(detections.wanted_detections)) + " detections")
    #     for my_class in detections.wanted_detections.keys():
    #         detection_count = 0
    #         detection_class_list = detections.wanted_detections[my_class]
    #         self.detection_patch_dict[my_class] = []
    #         for detection in detection_class_list:
    #             print("+++++++++++++++++++++++++++++++")
    #             print(detection_count)
    #             output_path = output_dir + detection.my_class + "/" + str(detection_count) + "/"
    #             detection_count += 1
    #
    #             box = detection.bounding_box
    #
    #             orig_patch = image[box.start_row: box.end_row, box.start_col: box. end_col]
    #             if save_patches:  # todo: fix if save patches == TRue
    #                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #                 cv2.imwrite(output_path + "original_patch.png", orig_patch)
    #
    #             patch_dict = {}
    #             for f_idx in range(len(self.frames)):
    #                 print(str((f_idx / len(self.frames)) * 100) + "%")
    #                 frame = self.frames[f_idx]
    #                 patch = frame[box.start_row:box.end_row, box.start_col:box.end_col]
    #                 score = image_euclid_dist(patch, orig_patch)
    #                 patch_dict[f_idx] = score
    #
    #             sorted_patches_frame_indices = sorted(patch_dict.keys(), key=lambda x: patch_dict[x])
    #             sorted_patch_alternatives = []
    #             for i in range(len(sorted_patches_frame_indices)):
    #                 f_idx = sorted_patches_frame_indices[i]
    #                 patch_score = patch_dict[f_idx]
    #                 patch_im = self.frames[f_idx][box.start_row:box.end_row, box.start_col:box.end_col]
    #                 sorted_patch_alternatives.append(PatchAlternative(patch_im, patch_score))
    #                 if save_patches:
    #                     cv2.imwrite(output_path + str(i) + ".png", patch_im)
    #             self.detection_patch_dict[my_class].append(PatchAlternativesCollection(orig_patch, sorted_patch_alternatives, box.start_row, box.start_col))
    #
    # def __optimize_patches(self, orig_image):
    #     print("=============================================")
    #     print("optimizing patches with " + str(int(self.used_close_patches_percentage * 100)) + "% of similiar patches...")
    #     count = 0
    #     for my_class in self.detection_patch_dict:
    #         # count = 0
    #         for patch_alternatives in self.detection_patch_dict[my_class]:
    #             used_alternatives = patch_alternatives.alternatives[:int(len(patch_alternatives.alternatives) * self.used_close_patches_percentage)]
    #             new_patch = sum(alternative.patch for alternative in used_alternatives) / len(used_alternatives)
    #             # new_patch = np.zeros(patch_alternatives.original_patch.shape)
    #             pixel_data_dict = {}
    #             # for row, col in ut.row_col_generator(patch_alternatives.alternatives[0].patch.shape[:2]):
    #                 # pixel_data = PixelData(row, col, list(range(len(patch_alternatives.alternatives))), [alternative.patch for alternative in patch_alternatives.alternatives])
    #                 # pixel_data_dict[(row, col)] = pixel_data
    #                 # new_patch[row, col] = pixel_data.colors[pixel_data.chosen_color_idx]
    #
    #             # cv2.imwrite(self.current_percentage_output_dir + str(count) + ".png", (new_patch* 255).astype('uint8'))  # todo: remove line
    #             count += 1
    #             orig_image[patch_alternatives.start_row: patch_alternatives.start_row + new_patch.shape[0], patch_alternatives.start_col: patch_alternatives.start_col + new_patch.shape[1]] = new_patch
    #     return orig_image  # this image is float: edit: now it is not float

# if __name__ == '__main__':
#     image_path = sys.argv[1]
#     video_path = sys.argv[2]
#     image = cv2.imread(image_path) / 255
#
#     f1 = cv2.imread('temp/nyc2/61_8_sim.png') / 255
#     f2 = cv2.imread('temp/nyc2/16_39_sim.png') / 255
#     f3 = cv2.imread('temp/nyc2/2_28_sim.png') / 255
#     f4 = cv2.imread('temp/nyc2/11_38_sim.png') / 255
#
#     frames = [f1, f2, f3, f4]
#     detector = yod.FrameDetector()
#
#     po = PatchOptimizer()
#
#     po.optimize_object_patches(image, frames, detector, save_similiar_patches=True)
