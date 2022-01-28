from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class FaceDetection(object):

    def __init__(self, model, image):

        self.model = model
        self.image = image

    def face_detection(self):
        boxes, _ = self.model.detect(self.image)
        frame_draw = self.image.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        plt.figure(figsize=(10,20))
        plt.imshow(frame_draw)
        plt.xticks([])
        plt.yticks([])
        plt.show()