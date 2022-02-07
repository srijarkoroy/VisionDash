import os
import re

import torch
import torchvision.transforms.functional as tvf
from torchvision import transforms

from PIL import Image
import gdown
import json

from algorithms.vgg import Vgg16
from algorithms.transformer_net import TransformerNet

class StyleTransfer:

    def __init__(self, style='candy'):
        '''
        Parameters:
        - image: PIL image input

        - noise: type of noise (text/gaussian)
        '''
        self.style = style

        if torch.cuda.is_available():
            self.map_location = 'cuda'

        else:
            self.map_location = 'cpu'
        
        try:
            self.model = TransformerNet()
            
        except Exception as err:
            print("Error at {}".format(err))
            exit()

        self.check_weights()
        self.load_model()
        # img.show()
        # self.inference(img)

    def check_weights(self):

        if os.path.exists("weights/{}.pth".format(self.style)):
            print("Found weights")

        else:
            print("Downloading weights")
            self.download_weights()

    def download_weights(self):
        with open("misc/weights_download.json") as fp:
            json_file = json.load(fp)
            if not os.path.exists("weights/"):
                os.mkdir("weights/")
            url = 'https://drive.google.com/uc?id={}'.format(json_file['{}.pth'.format(self.style)])
            gdown.download(url, "weights/{}.pth".format(self.style), quiet=False)
    
    def load_model(self):   

        with torch.no_grad():

            ckpt_dir = "weights/{}.pth".format(self.style)
            state_dict = torch.load(ckpt_dir, self.map_location)

            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]

            self.model.load_state_dict(state_dict)

    def transform_img(self, img):

        content_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])

        img = content_transform(img)
        return img

    def inference(self,img):

        img = self.transform_img(img)
        img = img.unsqueeze(0).to(self.map_location)

        with torch.no_grad():
            output = self.model(img)

        data = output[0].cpu()
        img = data.clone().clamp(0, 255).numpy()

        img = img.transpose(1, 2, 0).astype("uint8")
        styled= Image.fromarray(img)
        return styled

## Usage ##
# if __name__=="__main__":
#     img = Image.open('misc/testt.jpg')
#     model = StyleTransfer()
#     model.inference(img)




