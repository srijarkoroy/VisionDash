# MTCNN
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class FaceDetection(object):

    def __init__(self, model, image):

        '''
        Parameters:

        - model: pre-trained model to be used for face detection

        - image: PIL image input
        
        '''

        self.model = model
        self.image = image

    def face_detection(self):
        
        boxes, _ = self.model.detect(self.image)
        frame_draw = self.image.copy()
        draw = ImageDraw.Draw(frame_draw)
        
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
       
        return frame_draw

## Usage ##

#if __name__=="__main__":
#    model = MTCNN(keep_all=True)
#    img = Image.open("test.jpg")
#    m = FaceDetection(model, img)
#    m.detect()