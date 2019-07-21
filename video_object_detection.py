"""

"""
""" ============================ Imports ============================ """
from imutils.video import FPS
import imutils
import time
import cv2
import numpy as np
import my_utils as ut
# from multiprocessing.pool import ThreadPool
from threading import Thread
from threading import Lock

""" ============================ Constants ============================ """
FPD = 8  # frames per detection

""" ============================= Classes ============================= """


class VideoDetector:
    """

    """
    def __init__(self, frame_detector):
        self.frame_detector = frame_detector
        self.frame_detector.boarder_width = 1
        self.frame_detector.text_size = 0.3
        self.frame_detector.text_width = 1
        self.was_update = False
        self.tag_lock = Lock()

    @staticmethod
    def initialize_video_stream(filename):
        print('[INFO] starting video stream... [INFO]')
        if filename is None:
            vs = cv2.VideoCapture(0)
        else:
            vs = cv2.VideoCapture(filename)
        time.sleep(2.0)  # allowing sensor to worm up
        fps = FPS().start()
        return vs, fps

    @staticmethod
    def destruct_video_stream(vs, fps):
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        vs.release()
        cv2.destroyAllWindows()

    @staticmethod
    def update_boxes(tag_image, previous_boxes):
        new_boxes = []
        for box in previous_boxes:
            box_image = tag_image[box.start_row:box.end_row + 1, box.start_col:box.end_col + 1]
            negative_args = np.argwhere(box_image == -1)
            negative_args += [box.start_row, box.start_col]
            if len(negative_args) == 0:
                continue
            new_boxes.append(ut.Box(negative_args[0][0], negative_args[0][1],
                                    negative_args[-1][0], negative_args[-1][1]))
        return new_boxes

    def update_tag_image(self, tag_image, missing_boxes, frame, min_confidence, unwanted_objects, frame_index_count):
        was_update = False
        resized_frame = imutils.resize(frame, width=400)
        detections = self.frame_detector.detect_frame(resized_frame, unwanted_objects, min_confidence=min_confidence)
        detections.update_detections_to_image_coordinates(frame)
        for missing_box in missing_boxes:
            # for each missing box
            for row in range(missing_box.start_row, missing_box.end_row + 1):
                for col in range(missing_box.start_col, missing_box.end_col + 1):
                    # for each missing pixel of the box
                    if tag_image[row, col] > -1:
                        # if already found
                        continue
                    if not detections.is_pixel_inside_detections(row, col):
                        was_update = True
                        tag_image[row][col] = frame_index_count
        self.was_update = was_update

    def calculate_initial_frame_data(self, frame, tag_image, frames, unwanted_objects, min_confidence):
        resized_frame = imutils.resize(frame, width=400)
        detections = self.frame_detector.detect_frame(resized_frame, unwanted_objects, min_confidence=min_confidence)
        if detections.is_empty():
            return frame
        detections.update_detections_to_image_coordinates(frame)
        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                if not detections.is_pixel_inside_detections(row, col):
                    tag_image[row, col] = 0
        missing_boxes = detections.get_detected_boxes()
        frames.append(frame)
        return missing_boxes

    @staticmethod
    def quit_request_submitted():
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            return True
        return False

    @staticmethod
    def _construct_image(tag_image, frames):
        image = np.zeros((tag_image.shape[0], tag_image.shape[1], 3))
        for row in range(tag_image.shape[0]):
            for col in range(tag_image.shape[1]):
                if tag_image[row, col] == -1:
                    image[row, col] = frames[0][row, col]
                else:
                    image[row, col] = (frames[tag_image[row, col]])[row, col]
        return image.astype('uint8')

    def stream_detection(self, video_src, min_confidence=0.2, size_factor=1):

        print('[INFO] starting video stream... [INFO]')
        vs = self.initialize_video_stream(video_src)
        fps = FPS().start()

        # loop over the frames from the video stream
        while vs.isOpened():
            ret, frame = vs.read()
            # frame = imutils.resize(frame, width=400)

            detections = self.frame_detector.detect_frame(frame, min_confidence=min_confidence)

            ut.show_image("Video", frame, detections=detections, size_factor=size_factor, video=True)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # update the FPS counter
            fps.update()

        fps.stop()
        # stop the timer and display FPS information
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        self.destruct_video_stream(vs)

    def get_clean_image(self, video_src, unwanted_objects, min_confidence=0.2, size_factor=1):
        # initializing method variables
        vs, fps = self.initialize_video_stream(video_src)
        frames = []
        frame_index_count = 0
        frame_count = 0

        ret, frame = vs.read()
        tag_image = np.full((frame.shape[0], frame.shape[1]), -1)
        # first iteration
        missing_boxes = self.calculate_initial_frame_data(frame, tag_image, frames, unwanted_objects, min_confidence)
        ut.show_image("video", ut.draw_transparent_boxes(frame, missing_boxes), size_factor=size_factor, video=True)

        ret, frame = vs.read()
        frame_count += 1
        frame_index_count += 1

        # loop over the frames from the video stream
        # pool = ThreadPool(processes=1)
        while ret and len(missing_boxes) > 0:
            current_frame_on_detection = frame
            thread = Thread(target=self.update_tag_image, args=(tag_image, missing_boxes, frame, min_confidence,
                            unwanted_objects, frame_index_count))
            thread.start()

            while thread.is_alive() and ret:
                print("frame #{0}".format(str(frame_count)))
                ut.show_image("video", ut.draw_transparent_boxes(frame, missing_boxes), size_factor=size_factor,
                              video=True)
                frame_count += 1
                fps.update()
                if self.quit_request_submitted():
                    break
                ret, frame = vs.read()

            if self.was_update:
                self.was_update = False
                frames.append(current_frame_on_detection)
                frame_index_count += 1
                missing_boxes = self.update_boxes(tag_image, missing_boxes)

        self.destruct_video_stream(vs, fps)
        # reconstruct new image

        constructed_image = self._construct_image(tag_image, frames)
        ut.show_image("new", constructed_image)

    """ ============================ Not in use ============================ """

    def _unthreaded_get_clean_image(self, video_src, unwanted_objects, min_confidence=0.2, size_factor=1):
        # initializing method variables
        vs, fps = self.initialize_video_stream(video_src)
        frames = []
        frame_index_count = 0
        frame_count = 0

        ret, frame = vs.read()
        # new_shape = imutils.resize(frame, width=400).shape

        tag_image = np.full((frame.shape[0], frame.shape[1]), -1)

        # first iteration
        resized_frame = imutils.resize(frame, width=400)
        detections = self.frame_detector.detect_frame(resized_frame, unwanted_objects, min_confidence=min_confidence)
        if detections.is_empty():
            return frame
        detections.update_detections_to_image_coordinates(frame)
        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                if not detections.is_pixel_inside_detections(row, col):
                    tag_image[row, col] = frame_count
        missing_boxes = detections.get_detected_boxes()
        frames.append(frame)

        ret, frame = vs.read()
        frame_count += 1
        frame_index_count += 1

        ut.show_image("video", ut.draw_transparent_boxes(frame, missing_boxes), size_factor=size_factor, video=True)

        # loop over the frames from the video stream
        # while ret and pixel_count < (tag_image.shape[0] - 1) * (tag_image.shape[1] - 1):
        while ret and len(missing_boxes) > 0:
            print("frame #{0}".format(str(frame_count)))
            # grab the frame from the threaded video stream and resize it
            # to have a max width of 400 pixels

            if frame_count % FPD == 0:
                # detections = self.frame_detector.detect_frame(frame, min_confidence=min_confidence)
                was_update = self.update_tag_image(tag_image, missing_boxes, frame, min_confidence,
                                                   unwanted_objects, frame_index_count)
                if was_update:
                    frames.append(frame)
                    frame_index_count += 1
                    missing_boxes = self.update_boxes(tag_image, missing_boxes)

            ut.show_image("video", ut.draw_transparent_boxes(frame, missing_boxes),
                          size_factor=size_factor, video=True)

            frame_count += 1
            fps.update()
            if self.quit_request_submitted():
                break
            ret, frame = vs.read()

        self.destruct_video_stream(vs, fps)
        # reconstruct new image
        constructed_image = self._construct_image(tag_image, frames)
        ut.show_image("new", constructed_image)

    def unthreaded_update_tag_image(self, tag_image, missing_boxes, frame, min_confidence, unwanted_objects,
                                    frame_index_count):
        was_update = False
        resized_frame = imutils.resize(frame, width=400)
        detections = self.frame_detector.detect_frame(resized_frame, unwanted_objects, min_confidence=min_confidence)
        detections.update_detections_to_image_coordinates(frame)
        for missing_box in missing_boxes:
            # for each missing box
            for row in range(missing_box.start_row, missing_box.end_row + 1):
                for col in range(missing_box.start_col, missing_box.end_col + 1):
                    # for each missing pixel of the box
                    if tag_image[row, col] > -1:
                        # if already found
                        continue
                    if not detections.is_pixel_inside_detections(row, col):
                        was_update = True
                        tag_image[row][col] = frame_index_count
        return was_update


