"""
About this file:
"""
""" ============================ Imports ============================ """
import cv2
from mobilenet import mobilenet_object_detection as mod
from yolo import yolo_object_detection as yod
import video_object_detection as vod

""" ============================ Constants ============================ """
FILES_PATH = "files\\"
""" ============================ Functions ============================ """


if __name__ == '__main__':
    unwanted_objects = ["person"]
    image = cv2.imread(FILES_PATH + "garos.jpg")

    mobilenet_image_detector = mod.FrameDetector()
    yolo_image_detector = yod.FrameDetector()

    video_detector = vod.VideoDetector(yolo_image_detector)

    # detections = yolo_image_detector.detect_frame(image, min_confidence=0.2)
    # ut.show_image("image", image, detections=detections)

    video_detector.get_clean_image(FILES_PATH + "wimbeldon2.mp4", unwanted_objects, min_confidence=0.5)
    # video_detector.stream_detection(FILES_PATH + "met2.mp4", min_confidence=0.5, size_factor=2)