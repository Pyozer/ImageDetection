
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()

camera = cv2.VideoCapture("http://192.168.43.41:8080/video")

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel("fastest")

custom_objects = detector.CustomObjects(car=True, motorcycle=True, truck=True)

video_path = detector.detectCustomObjectsFromVideo(
    custom_objects=custom_objects,
    #input_file_path=os.path.join(execution_path, "traffic.mp4"),
    camera_input=camera,
    output_file_path=os.path.join(execution_path, "traffic_detected"),
    frames_per_second=17,
    minimum_percentage_probability=5,
    frame_detection_interval=2,
    log_progress=True
)