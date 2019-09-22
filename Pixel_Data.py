
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
