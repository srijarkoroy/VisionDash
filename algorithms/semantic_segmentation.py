# DeepLabV3
import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import random

class SemanticSegmentation(object):

    def __init__(self, model, image):

        '''
        Parameters:

        - model: pre-trained model to be used for semantic segmentation

        - image: PIL image input
        
        '''

        self.model = model.eval()
        self.image = image

    # Define the helper function
    def segmap(self, img, nc=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        
        r = np.zeros_like(img).astype(np.uint8)
        g = np.zeros_like(img).astype(np.uint8)
        b = np.zeros_like(img).astype(np.uint8)

        for l in range(0, nc):

            idx = img == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def semantic_segmentation(self):

        #img = Image.open(path)
        #plt.imshow(img); plt.axis('off'); plt.show()
        # Comment the Resize and CenterCrop for better inference results
        trf = T.Compose([T.Resize(256),
                        T.ToTensor(), 
                        T.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])
        inp = trf(self.image).unsqueeze(0)
        out = self.model(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        rgb = self.segmap(om)
        
        #plt.figure(figsize=(10,20))
        #plt.imshow(rgb); plt.axis('off'); plt.show()

        return Image.fromarray(rgb)
        
## Usage ##

#if __name__=="__main__":
#    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=1)
#    img = Image.open("test.jpg")
#    m = SemanticSegmentation(model, img)
#    m.detect()