""" ============================ Imports =========================== """
import my_utils as ut
from Quantization import *
import numpy as np

""" ============================ Class ============================ """


class PixelData:

    def __init__(self, row, col, background_map, frames):
        self.row, self.col = row, col
        self.colors = None
        self.idx_to_cluster_size_dict = None
        self.sorted_idx_cluster_list = None
        self.__calculate_pixel_data(background_map, frames)
        self.chosen_color_idx = self.sorted_idx_cluster_list[0][0]
        self.color_max_distance = self.__calculate_color_max_distance()
        self.is_pure_background = (len(background_map) == len(frames))

    def get_largest_cluster_size(self):
        return self.sorted_idx_cluster_list[0][1]

    def get_used_cluster_size(self):
        return self.idx_to_cluster_size_dict[self.chosen_color_idx]

    def get_chosen_color_idx(self):
        return self.chosen_color_idx

    # def get_cluster_size_by_frame(self, frame_idx):
    #     return self.frame_to_cluster_size_dict[frame_idx]

    def get_color_by_idx(self, color_idx):
        return self.colors[color_idx]

    def get_color_value(self):
        return self.colors[self.chosen_color_idx]

    def __calculate_pixel_data(self, background_map, frames):
        """
        Creates a PixelData object from video data gathered.
        :param row: row of the pixel
        :param col: col of the pixel
        :param background_map: the mapping of the frames which represent a background value for this pixel
        :param frames: all frames gathered from the video.
        :return: PixelData object of the given row, col pixel
        """
        if len(background_map) == len(frames):
            sorted_frame_cluster_list = [(k, len(frames)) for k in range(len(frames))]
        else:
            pixel_clusters = self.__calculate_pixel_clusters(background_map, frames)
            sorted_frame_cluster_list = [(k, len(pixel_clusters[k])) for k in
                                         sorted(pixel_clusters.keys(), key=lambda k: len(pixel_clusters[k]), reverse=True)]
        self.idx_to_cluster_size_dict, self.sorted_idx_cluster_list = \
            self.__quantize_pixel_data_frame_options(frames, 4, 3, sorted_frame_cluster_list)

    def __quantize_pixel_data_frame_options(self, frames, n_quant, n_iter, sorted_frame_cluster_list):
        # creating a color list
        pixel = self.row, self.col
        colors = np.array([frames[f][pixel] for f in [t[0] for t in sorted_frame_cluster_list]])

        # calculating k-means
        self.colors, quants, error = quantize_rgb_list(colors, n_quant, n_iter)
        idx_to_cluster_size_dict = {i: 0 for i in range(len(self.colors))}
        for i in range(len(quants)):
            idx_to_cluster_size_dict[quants[i]] += 1
        sorted_idx_cluster_list = [(c, idx_to_cluster_size_dict[c]) for c in idx_to_cluster_size_dict.keys()]
        sorted_idx_cluster_list.sort(key=lambda x: x[1], reverse=True)

        return idx_to_cluster_size_dict, sorted_idx_cluster_list

    def __calculate_pixel_clusters(self, background_map, frames):
        """
        Calculates a dictionary of {pixel: cluster} here pixel is the frame number the pixel was taken from and
        cluster is all other frames which their pixels are close up to a threshold to the key pixel.
        :param row: row of the pixel
        :param col: col of the pixel
        :param background_map: the mapping of the frames which represent a background value for this pixel
        :param frames: all frames gathered from the video.
        :return: A dictionary of pixels and their clusters
        """
        background_frame_indices = background_map  # a list of frames this pixel was in the background
        pixel_clusters = {k: [k] for k in background_frame_indices}
        for i in range(len(background_frame_indices)):
            for j in range(i + 1, len(background_frame_indices)):
                frame_index = background_frame_indices[i]
                other_index = background_frame_indices[j]
                if ut.euclidean_dist(frames[frame_index][self.row, self.col],
                                     frames[other_index][self.row, self.col]) <= ut.PIXEL_SIMILIAR_THRESHOLD:
                    pixel_clusters[frame_index].append(other_index)
                    pixel_clusters[other_index].append(frame_index)
        return self.__process_unique_pixel_clusters(pixel_clusters, frames)

    def __process_unique_pixel_clusters(self, pixel_clusters, frames):
        filtered_dict = {}
        seen_color_values = []

        for key in pixel_clusters:
            color_tup = tuple(frames[key][self.row, self.col])
            if color_tup not in seen_color_values:
                filtered_dict[key] = pixel_clusters[key]
                seen_color_values.append(color_tup)
        return filtered_dict

    def __calculate_color_max_distance(self):
        max_dist = 0
        if len(self.colors > 1):
            for i in range(len(self.colors)):
                for j in range(i, len(self.colors)):
                    curr_dist = ut.euclidean_dist(self.colors[i], self.colors[j])
                    if curr_dist > max_dist:
                        max_dist = curr_dist
        return max_dist
