"""

"""
""" ============================ Imports ============================ """
from imutils.video import FPS
import time
import cv2
import numpy as np
import my_utils as ut
from yolo import yolo_object_detection as yod
from Pixel_Data import *
from Graph_Optimizer import *
from Patch_Optimization import *
import Quantization


""" ============================ Constants ============================ """
FPD = 8  # frames per detection
MIN_CONFIDENCE = 0.05
DETECTION_RESIZE_FACTOR = -0.1
USED_CLOSE_PATCH_PERCENTAGE=0.15
UNIFY_COLLISION = False
STREAM_SIZE_FACTOR = 1
SHOW_VIDEO = False

QUANTIZE_IMAGE = False
QUANTIZATION_COLOR_COUNT = 100
QUANTIZATION_ITERS = 3

QUANTIZE_FRAME_WIDE = False


""" ============================= Classes ============================= """


class VideoDetector:
    """
    An objects which can detect object on video streams
    """
    def __init__(self, frame_detector):
        self.frame_detector = frame_detector
        self.frame_detector.boarder_width = 1
        self.frame_detector.text_size = 0.3
        self.frame_detector.text_width = 1
        self.was_update = False
        self.missing_pixels = []
        self.frame_index_count = 0
        self.pixel_background_frames_map = None
        self.frames = None
        self.pixel_data_dict = None
        self.optimizer = None
        self.output_name_folder = ""
        self.output_path = ""
        self.alpha = 0
        self.file_name = "No_Name"

    # ===================== General Class Methods =====================

    def __initialize_video_stream(self, filename):
        """
        Initializes a video stream
        :param filename: the path to the video file, if None is given use web-cam
        :return: vs and fps objects
        """
        ut.log_or_print('[INFO] starting video stream... [INFO]')
        if filename is None:
            vs = cv2.VideoCapture(0)
        else:
            vs = cv2.VideoCapture(filename)
        time.sleep(2.0)  # allowing sensor to worm up
        fps = FPS().start()
        return vs, fps

    def __destruct_video_stream(self, vs, fps):
        """
        Closes the video stream.
        """
        fps.stop()
        ut.log_or_print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        ut.log_or_print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        vs.release()
        cv2.destroyAllWindows()

    @staticmethod
    def quit_request_submitted():
        """ checks if the q button was pressed"""
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            return True
        return False

    def refresh(self, output_name_folder, alpha):
        self.alpha = alpha
        self.output_name_folder = output_name_folder
        self.output_path = self.output_name_folder + str((int(alpha * 100))) + '/'
        if self.optimizer is not None:
            self.optimizer.refresh(alpha)

    # ===================== Video Detection Methods =====================

    def object_sensitive_video_merge(self, video_src, unwanted_objects):
        """
        Receives a video path and unwanted objects list and returns a clean image.
        :param video_src: the path for the video
        :param unwanted_objects: a list of object we want to clean from the image
        :return:
        """
        # Analyzing raw image data
        if self.pixel_background_frames_map is None or self.frames is None:
            self.__analyze_video_data(video_src, unwanted_objects, detection_resize_factor=DETECTION_RESIZE_FACTOR)

        median_image = ut.get_image_median(self.frames, self.pixel_background_frames_map)
        # average_image = ut.get_image_average(self.frames, self.pixel_background_frames_map)
        # cv2.imwrite("average.png", average_image)
        # return

        # median_image = cv2.imread("10.png")


        # # todo return this code back for normal flow, for debuggin leave unmarked.
        # # Processing video data
        # if self.pixel_data_dict is None:
        #     self.__process_video_data()
        #
        # # Initialize Graph optimization algorithm with processed data
        # if self.optimizer is None:
        #     self.optimizer = GraphOptimizer(self.pixel_data_dict, self.frames, self.alpha)
        #
        # # Run optimization process
        # merged_image = self.optimizer.optimize_input(self.output_path)  # at this point, merged_image is float
        # merged_image = (merged_image * 255).astype('uint8')
        # cv2.imwrite(self.output_name_folder + str(int(self.alpha * 100)) + "graph_optimized.png", merged_image)
        # # merged_image = cv2.imread("temp/output.png")

        # merged_image = cv2.imread("10.png")

        # optimize patches of wanted objects
        patch_optimizer = PatchOptimizer(self.frame_detector, self.frames, self.file_name, used_close_patches_percentage=USED_CLOSE_PATCH_PERCENTAGE)
        # optimize unwanted objects
        # merged_image = patch_optimizer.optimize_unwanted_objects(merged_image)
        # optimize patches of wanted objects
        merged_image = patch_optimizer.optimize_wanted_object_patches(median_image, self.pixel_background_frames_map)  # float image, edit: now it is not
        # optimize patches of background pixels
        # patch_optimizer.optimize_background_patches(merged_image)
        cv2.imwrite("output/" + self.file_name + "/" + str(0.5 * 100) + ".png", merged_image)

    # def get_pixels_to_refine(self, image):
    #     detections = self.frame_detector.detect_frame(image, ['person'], detection_resize_factor=0.15)
    #     pixel_batches = []
    #     for detection_list in detections.unwanted_detections.values():
    #         for detection in detection_list:
    #             box = detection.bounding_box
    #             start_row, end_row, start_col, end_col = box.start_row, box.end_row, box.start_col, box.end_col
    #             patch = image[start_row:end_row, start_col:end_col]
    #             # cv2.imwrite("patch.png", patch)
    #             background_color = (patch[0][0] + patch[0][-1] + patch[-1][0] + patch[-1, -1]) // 4
    #             background_color_image = np.full((500, 500, 3), background_color)
    #             distance_from_background = np.zeros(patch.shape[:2])
    #             for i in range(patch.shape[0]):
    #                 for j in range(patch.shape[1]):
    #                     distance_from_background[i, j] = ut.euclidean_dist(background_color.astype('int'),
    #                                                                        patch[i, j].astype('int'))
    #             distance_from_background = (distance_from_background * 255 / np.max(distance_from_background)).astype(
    #                 'uint8')
    #             thresh = np.mean(distance_from_background)
    #             print(thresh)

    def __analyze_video_data(self, video_src, unwanted_objects, detection_resize_factor=DETECTION_RESIZE_FACTOR, show_video=SHOW_VIDEO):
        ut.log_or_print("[INFO] Starting Video Analyzation... [INFO]")
        start_analyze = time.time()
        vs, fps = self.__initialize_video_stream(video_src)
        total_frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        analized_frames_count = (total_frame_count // FPD) + 1
        ut.log_or_print("[INFO] Analyzing " + str(analized_frames_count) + " out of " + str(total_frame_count) + " frames")

        self.frames = []
        frame_count = 0
        ret, frame = vs.read()

        # a dictionary which holds for every pixel its background frame indices
        pixel_keys = [(i, j) for i in range(frame.shape[0]) for j in range(frame.shape[1])]
        self.pixel_background_frames_map = {k: [] for k in pixel_keys}

        # counts the indices of frames added to 'frames array
        frame_index_count = 0

        while ret:
            if frame_count % FPD == 0:
                ut.log_or_print("=================================================")
                ut.log_or_print("[INFO] Frame: " + str(frame_index_count + 1))
                ut.log_or_print("[INFO] Percentage Done: " + str(((frame_index_count + 1) / analized_frames_count) * 100))

                detections = self.frame_detector.detect_frame(frame, unwanted_objects, min_confidence=MIN_CONFIDENCE, detection_resize_factor=detection_resize_factor)

                if QUANTIZE_IMAGE:  # perform RGB quantization on frame before storage
                    quantization_output = Quantization.quantize_rgb_image(frame, QUANTIZATION_COLOR_COUNT, QUANTIZATION_ITERS)
                    frame = quantization_output[0].astype('uint8')

                start = time.time()
                # get a binary map which shows where there is an unwanted detection
                detection_map = detections.get_detection_map(frame.shape, unify_collisions=UNIFY_COLLISION)
                for row in range(detection_map.shape[0]):
                    for col in range(detection_map.shape[1]):
                        if detection_map[row, col] == 0:
                            self.pixel_background_frames_map[(row, col)].append(frame_index_count)
                end = time.time()

                ut.log_or_print("[INFO] updating pixel info took {:.6f} seconds".format(end - start))
                self.frames.append(frame / 255)
                frame_index_count += 1

            if show_video:
                ut.show_image(frame, size_factor=STREAM_SIZE_FACTOR, video=True)
            fps.update()
            # if self.quit_request_submitted():
            #     break
            ret, frame = vs.read()
            frame_count += 1

        self.__destruct_video_stream(vs, fps)
        end_analyze = time.time()
        ut.log_or_print("=================================================")
        ut.log_or_print("[INFO] Video analyzing ended after {:.6f} seconds".format(end_analyze - start_analyze))

    def __process_video_data(self):
        ut.log_or_print("[INFO] Processing video data...")
        start = time.time()
        self.pixel_data_dict = {}
        ut.log_or_print("[INFO] Initializing pixels data structures...")
        for row, col in ut.row_col_generator(self.frames[0].shape[:2]):

            # if no background frames were found for this pixel check all frames as background
            if len(self.pixel_background_frames_map[row, col]) == 0:
                self.pixel_background_frames_map[row, col] = list(range(len(self.frames)))

            pixel_data = PixelData(row, col, self.pixel_background_frames_map[(row, col)], self.frames)
            self.pixel_data_dict[(row, col)] = pixel_data
        end = time.time()
        ut.log_or_print_time(start, end)

    """ ============================ Streaming Functions ============================ """

    def stream_detection(self, video_src, unwanted_objects):
        """
        Stream the video detection of unwanted objects in a given video source
        :param video_src: the path for the video
        :param unwanted_objects: a list of unwanted objects
        :return:
        """
        vs, fps = self.__initialize_video_stream(video_src)

        # loop over the frames from the video stream
        ret, frame = vs.read()
        while ret:

            detections = self.frame_detector.detect_frame(frame, min_confidence=MIN_CONFIDENCE, unwanted_objects=unwanted_objects)

            if QUANTIZE_IMAGE:
                quantization_output = Quantization.quantize_rgb_image(frame, QUANTIZATION_COLOR_COUNT, QUANTIZATION_ITERS)
                frame = quantization_output[0].astype('uint8')
            ut.show_image(frame, detections=detections, size_factor=STREAM_SIZE_FACTOR, video=True)

            # if the `q` key was pressed, break from the loop
            if self.quit_request_submitted():
                break

            # update the FPS counter
            fps.update()
            ret, frame = vs.read()

        self.__destruct_video_stream(vs, fps)
