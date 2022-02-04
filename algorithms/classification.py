# Efficientnet

from torchvision import models
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class Classification(object):

    def __init__(self, model, image):

        '''

        Parameters:

        - model: pre-trained model to be used for image classification

        - image: PIL image input
        
        '''

        self.model = model.eval()
        self.image = image


    def preprocessImg(self):
        transform = transforms.Compose([ 
        transforms.Resize(256), 
        transforms.CenterCrop(224),  
        transforms.ToTensor(),  
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]  
        )])
        img = transform(self.image)
        img_t = torch.unsqueeze(img, 0)
        return img_t
    
    def classification(self):
        self.output = self.model(self.preprocessImg())

        with open('./misc/simple_imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        _, index = torch.max(self.output, 1)
        img = np.array(self.image) 
        
        #plt.imshow(img)
        #plt.show()
        #print(classes[index[0]])

        return classes[index[0]]


## Usage ##

# if __name__=="__main__":
#     model = models.efficientnet_b7(pretrained=True)
#     img = Image.open("test2.jpg")
#     m = Classification(model, img)
#     m.detect()

