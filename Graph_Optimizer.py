""" ============================ Imports ============================ """
import my_utils as ut
import numpy as np
import time
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


""" ============================ Constants ============================ """
ITERS = 1000
OUTPUT_MID_OPTIMIZED = True

""" ============================ Class ============================ """


class GraphOptimizer:

    def __init__(self, pixel_data_dict, frames, alpha=1000):
        # receiving arguments
        self.pixel_data_dict = pixel_data_dict
        self.frames = frames
        self.shape = frames[0].shape
        self.alpha = alpha

        # initial data calculations
        self.graph = {(i, j): GraphNode() for i, j in ut.row_col_generator(self.shape, print_percentage=False)}
        self.__initialize_graph_edges()

        # calculating initial error values
        self._total_err = 0
        self.__calculate_initial_graph_error()

    def refresh(self, alpha):
        self.alpha = alpha
        self._total_err = 0
        self.graph = {(i, j): GraphNode() for i, j in ut.row_col_generator(self.shape, print_percentage=False)}
        self.__initialize_graph_edges()
        self.__calculate_initial_graph_error()

    def optimize_input(self, output_path, num_of_iterations=ITERS):
        ut.log_or_print("===================================================")
        ut.log_or_print("[INFO] Starting Optimization Process")
        start = time.time()
        iteration = 0
        changed = True
        all_dirty = True
        pixel_data_dict_copy = copy.deepcopy(self.pixel_data_dict)

        iteration_error_rates = []
        curr_dirty_pixels = set()
        while iteration < num_of_iterations and changed:
            # initializing iteration base info
            iter_start = time.time()
            changed = False
            changed_pixel_count = 0
            ut.log_or_print("===================================================")
            ut.log_or_print("iteration " + str(iteration + 1) + ":")

            if all_dirty:
                dirty_pixels = ut.row_col_generator(self.shape, print_percentage=True)
                all_dirty = False
                ut.log_or_print("[INFO] Iteration over all pixels")
            else:
                dirty_pixels = curr_dirty_pixels
                curr_dirty_pixels = set()

            for pixel in dirty_pixels:
                # initializing pixel base info
                row, col = pixel
                pixel_node = self.graph[pixel]
                pixel_data = pixel_data_dict_copy[(row, col)]
                local_color_err = pixel_node.sum_edge_values()
                local_cluster_size_err = pixel_node.value
                pixel_options = pixel_data.sorted_idx_cluster_list

                initial_local_error = self.alpha * local_color_err + local_cluster_size_err
                best_pixel_color_idx = pixel_data.chosen_color_idx
                curr_best_local_err = initial_local_error
                local_change = False
                for color_idx, cluster_size in pixel_options:
                    # trying to find a new better frame
                    if color_idx != pixel_data.chosen_color_idx:
                        # calculate new cluster size err
                        cluster_size_local_err = abs(pixel_data.get_largest_cluster_size() - cluster_size)

                        # calculate edges errors
                        neighbours = pixel_node.edges.keys()
                        local_color_err = 0
                        for neighbor in neighbours:
                            local_color_err += ut.euclidean_dist(pixel_data.get_color_by_idx(color_idx),
                                                                 pixel_data_dict_copy[neighbor].get_color_value())
                        new_total_err = self.alpha * local_color_err + cluster_size_local_err
                        if new_total_err < curr_best_local_err:
                            local_change = True
                            curr_best_local_err = new_total_err
                            best_pixel_color_idx = color_idx
                            changed_pixel_count += 1

                if local_change:
                    changed = True
                    # update pixel info
                    pixel_data_dict_copy[pixel].chosen_color_idx = best_pixel_color_idx
                    self.graph[pixel].value = abs(pixel_data.get_largest_cluster_size() - pixel_data.get_used_cluster_size())
                    for neighbor in self.graph[pixel].edges.keys():
                        color_err_value = ut.euclidean_dist(pixel_data.get_color_value(),
                                                            pixel_data_dict_copy[neighbor].get_color_value())
                        self.graph[pixel].edges[neighbor] = color_err_value
                        self.graph[neighbor].edges[pixel] = color_err_value
                        curr_dirty_pixels.add(neighbor)
                    self._total_err -= (initial_local_error - curr_best_local_err)
                    ut.log_or_print("Optimized pixel (" + str(row) + ", " + str(col) + "); Curr error: " + str(self._total_err), log_msg=False, print_msg=False)
            iteration += 1
            iter_end = time.time()
            iteration_error_rates.append(self._total_err)
            ut.log_or_print_time(iter_start, iter_end)
            ut.log_or_print("[INFO] Number of changed pixels: " + str(changed_pixel_count))
            ut.log_or_print("[INFO] Current error rate: " + str(self._total_err))
            if OUTPUT_MID_OPTIMIZED:
                image = (self.__construct_optimized_image() * 255).astype('uint8')
                cv2.imwrite(output_path + str(self.alpha) + "_" + str(iteration) + "_output.png", image)

        end = time.time()
        ut.log_or_print("[INFO] Optimization complete after " + str(iteration) + " iterations")
        ut.log_or_print_time(start, end)
        try:
            plt.plot(list(range(len(iteration_error_rates))), iteration_error_rates)
            plt.axis([0, len(iteration_error_rates), iteration_error_rates[-1] - 200, iteration_error_rates[0] + 200])
            plt.savefig(output_path + "itreation_error_rates.png")
        except Exception as e:
            ut.log_or_print("Exception Accure while trying to draw plot:\n")
        return self.__construct_optimized_image()

    def __calculate_initial_graph_error(self):
        ut.log_or_print("===================================================")
        start = time.time()
        ut.log_or_print("[INFO] Initializing image graph data...")
        color_diff_err = 0
        cluster_size_diff_err = 0

        for row, col in ut.row_col_generator(self.shape, print_percentage=True):
            # calculate error value for distance from best cluster size
            pixel_data = self.pixel_data_dict[(row, col)]
            cluster_size_err = abs(pixel_data.get_largest_cluster_size() - pixel_data.get_used_cluster_size())
            self.graph[(row, col)].value = cluster_size_err
            cluster_size_diff_err += cluster_size_err

            # calculate error values for distance of color from neighbours
            if row < self.shape[0] - 1:
                # evaluate neighbour under pixel
                neighbour_data = self.pixel_data_dict[(row + 1, col)]
                color_err = ut.euclidean_dist(pixel_data.get_color_value(), neighbour_data.get_color_value())
                self.graph[(row, col)].edges[(row + 1, col)] = color_err
                self.graph[(row + 1, col)].edges[(row, col)] = color_err
                color_diff_err += color_err
            if col < self.shape[1] - 1:
                # evaluate neighbour to the right of pixel
                neighbour_data = self.pixel_data_dict[(row, col + 1)]
                color_err = ut.euclidean_dist(pixel_data.get_color_value(), neighbour_data.get_color_value())
                self.graph[(row, col)].edges[(row, col + 1)] = color_err
                self.graph[(row, col + 1)].edges[(row, col)] = color_err
                color_diff_err += color_err
        self._total_err = self.alpha * color_diff_err + cluster_size_diff_err
        end = time.time()
        ut.log_or_print_time(start, end)
        ut.log_or_print("Current Error: " + str(self._total_err))

    def __initialize_graph_edges(self):
        for row, col in ut.row_col_generator(self.shape, print_percentage=False):
            if row < self.shape[0] - 1:
                self.graph[(row, col)].edges[(row + 1, col)] = 0
                self.graph[(row + 1, col)].edges[(row, col)] = 0
            if col < self.shape[1] - 1:
                self.graph[(row, col)].edges[(row, col + 1)] = 0
                self.graph[(row, col + 1)].edges[(row, col)] = 0

    def __construct_optimized_image(self):
        ut.log_or_print("=================================================")
        ut.log_or_print("[INFO] Constructing optimized image...")
        ut.log_or_print("[INFO] Error Rate: " + str(self._total_err))
        image = np.zeros((self.shape[0], self.shape[1], 3))
        for row, col in ut.row_col_generator(self.shape, print_percentage=False):
            pixel_data = self.pixel_data_dict[(row, col)]
            image[row, col] = pixel_data.get_color_value()
        return image


class GraphNode:

    def __init__(self):
        self.value = 0
        self.edges = {}

    def sum_edge_values(self):
        return np.sum(list(self.edges.values()))

