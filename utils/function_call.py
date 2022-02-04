import torch
import torchvision
from facenet_pytorch import MTCNN

from algorithms.classification import Classification
from algorithms.face_detection import FaceDetection
from algorithms.objectDetection import ObjectDetection
from algorithms.instance_segmentation import InstanceSegmentation
from algorithms.semantic_segmentation import SemanticSegmentation

# Classification
def classify(image):

    model = torchvision.models.efficientnet_b7(pretrained=True)
    classifier = Classification(model, image)
    return classifier.classification()


# Face Detection
def face_detect(image):

    model = MTCNN(keep_all=True)
    face_detector = FaceDetection(model, image)
    return face_detector.face_detection()


# Object Detection
def object_detect(image):

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    object_detector = ObjectDetection(model, image)
    return object_detector.detect()


# Instance Segmentation
def instance_segment(image):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    instance_segmentor = InstanceSegmentation(model, image)
    return instance_segmentor.instance_segmentation()


# Semantic Segmentation
def semantic_segment(image):

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=1)
    semantic_segmentor = SemanticSegmentation(model, image)
    return semantic_segmentor.semantic_segmentation()