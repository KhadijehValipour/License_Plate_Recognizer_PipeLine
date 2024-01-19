import os
from ultralytics import YOLO
from deep_text_recognition_benchmark.dtrb import DTRB

plate_detector = YOLO("weights\yolov8_detector\yolov8-x-license-plate-detector.pt")
plate_detector.predict("io\input\image1.jpg" , save=True , save_crop=True)

# plate_detector.predict("io\InputPlates\IMG42.jpg" , save=True , save_crop=True)


plate_recognizer = DTRB("weights\dtrb_recognizer\Dtrb_TPS-ResNet-BiLSTM-Attn_License_Plate_Recognizer.pth")
plate_recognizer.predict("runs\detect\predict3\crops")


