import cv2
import argparse
import string
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from deep_text_recognition_benchmark.dtrb import DTRB




parser = argparse.ArgumentParser()
# parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
# parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default="Attn", help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
parser.add_argument('--detector-weights', type=str, default="weights\yolov8_detector\yolov8-x-license-plate-detector.pt")
parser.add_argument('--recognizer_weights', type=str, default="weights\dtrb_recognizer\Dtrb_TPS-ResNet-BiLSTM-Attn_License_Plate_Recognizer.pth")
parser.add_argument('--input_image', type=str, default="io\input\image3.jpg")
parser.add_argument('--threshold', type=float, default=0.5)
opt = parser.parse_args()


""" vocab / character number configuration """
if opt.sensitive:
    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()


image = cv2.imread(opt.input_image)

plate_detector = YOLO(opt.detector_weights)
plate_recognizer = DTRB(opt.recognizer_weights, opt)
results = plate_detector.predict(image)
results = results[0]
for i, result in enumerate(results): # It will be executed as many images as it is going to detect their license plates
    for j in range(len(result.boxes.xyxy)): 
        if result.boxes.conf[j] > opt.threshold:
            bbox_tensor = result.boxes.xyxy[j]
            bbox_ndarray = bbox_tensor.cpu().detach().numpy().astype(int)
            print(bbox_ndarray)
            x, y, w, h = bbox_ndarray[0], bbox_ndarray[1], bbox_ndarray[2], bbox_ndarray[3]
            plate_image = image[y:h , x:w].copy()
            plate_image = cv2.resize(plate_image, (100, 32))
            plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"io\output\plate_{i}_{j}.jpg", plate_image)
            cv2.rectangle(image, (x, y), (w, h), (0,255,0), 4)
            print(plate_image)
            
            plate_recognizer.predict(plate_image, opt)
        

cv2.imwrite("io\output\image_result_1.jpg", image)


# plate_detector.predict("io\InputPlates\IMG42.jpg" , save=True , save_crop=True)



