""" ============================ Imports ============================ """
import my_utils as ut

""" ============================ Class ============================ """


class PixelData:

    def __init__(self, row, col, source_frame, frame_to_cluster_size_dict, sorted_frame_cluster_list):
        self.row, self.col = row, col
        self.source_frame = source_frame
        self.frame_to_cluster_size_dict = frame_to_cluster_size_dict
        self.sorted_frame_cluster_list = sorted_frame_cluster_list

    def get_largest_cluster_size(self):
        return self.sorted_frame_cluster_list[0][1]

    def get_used_cluster_size(self):
        return self.frame_to_cluster_size_dict[self.source_frame]

    def get_source_frame(self):
        return self.source_frame

    def get_cluster_size_by_frame(self, frame_idx):
        return self.frame_to_cluster_size_dict[frame_idx]

    def get_color_value(self, frames):
        return frames[self.source_frame][self.row, self.col]


""" ============================ Public Functions ============================ """


def calculate_pixel_data(row, col, background_map, frames):
    """
    Creates a PixelData object from video data gathered.
    :param row: row of the pixel
    :param col: col of the pixel
    :param background_map: the mapping of the frames which represent a background value for this pixel
    :param frames: all frames gathered from the video.
    :return: PixelData object of the given row, col pixel
    """
    pixel_clusters = __calculate_pixel_clusters(row, col, background_map, frames)
    sorted_frame_cluster_list = [(k, len(pixel_clusters[k])) for k in
                                 sorted(pixel_clusters.keys(), key=lambda k: len(pixel_clusters[k]), reverse=True)]
    frame_to_cluster_size_dict = {p[0]: p[1] for p in sorted_frame_cluster_list}
    source_frame = sorted_frame_cluster_list[0][0]
    return PixelData(row, col, source_frame, frame_to_cluster_size_dict, sorted_frame_cluster_list)


""" ============================ Private Functions ============================ """


def __calculate_pixel_clusters(row, col, background_map, frames):
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
            if ut.euclidean_dist(frames[frame_index][row, col],
                              frames[other_index][row, col]) <= ut.PIXEL_SIMILIAR_THRESHOLD:
                pixel_clusters[frame_index].append(other_index)
                pixel_clusters[other_index].append(frame_index)
    return __process_unique_pixel_clusters(pixel_clusters, row, col, frames)


def __process_unique_pixel_clusters(pixel_clusters, row, col, frames):
    filtered_dict = {}
    seen_color_values = []

    for key in pixel_clusters:
        color_tup = tuple(frames[key][row, col])
        if color_tup not in seen_color_values:
            filtered_dict[key] = pixel_clusters[key]
            seen_color_values.append(color_tup)
    return filtered_dict
