# retinaNet
from torchvision import models
import torchvision.transforms as T
import torch
from PIL import Image, ImageDraw

__COCO_INSTANCE_CATEGORY_NAMES__ = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ObjectDetection(object):
    
    def __init__(self, model, image):

        '''
        Parameters:

        - model: pre-trained model to be used for object detection

        - image: PIL image input
        
        '''

        self.image = image
        self.model = model.eval()

    def imagePreprocess(self):
        transform = T.Compose([T.ToTensor()])
        return torch.unsqueeze(transform(self.image),0)

    def object_detection(self):
        img_t = self.imagePreprocess()
        outp = self.model(img_t)
        return outp

    def detect(self):
        output = self.object_detection()
        bboxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(output[0]['boxes'].detach().numpy())]
        labels = [__COCO_INSTANCE_CATEGORY_NAMES__[i] for i in list(output[0]['labels'].numpy())]
        scores = list(output[0]['scores'].detach().numpy())
        finalPred = [scores.index(i) for i in scores if i>0.8][-1]
        boxes = bboxes[:finalPred+1]

        for i in range(len(boxes)):
            img1 = ImageDraw.Draw(self.image)  
            img1.rectangle([(boxes[i][0]),(boxes[i][1])], outline ="blue", width=3)

        return self.image


## Usage ##

#if __name__=="__main__":
#    model = models.detection.retinanet_resnet50_fpn(pretrained=True)
#    img = Image.open("test.jpg")
#    m = objectDetection(model, img)
#    m.detect()
    
