# N2N
import os
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
import gdown
import json
from unet import Unet

class Noise2Noise:

    def __init__(self,img,noise='text'):
        '''
        Parameters:

        - model: pre-trained model to be used for noise2noise

        - image: PIL image input

        - noise: type of noise (text/gaussian)
        '''
        self.noise = noise
        self.img = img

        if torch.cuda.is_available():
            self.map_location = 'cuda'
        else:
            self.map_location = 'cpu'
        
        try:
            self.model = Unet(in_channels=3)
            
        except Exception as err:
            print("Error at {}".format(err))
            exit()

        self.check_weights()
        self.load_model()
        # img.show()
        # self.inference(img)

    def check_weights(self):
        if os.path.exists("weights/n2n-{}.pt".format(self.noise)):
            print("Found weights")
        else:
            print("Downloading weights")
            self.download_weights()

    def download_weights(self):
        with open("misc/weights_download.json") as fp:
            json_file = json.load(fp)
            if not os.path.exists("weights/"):
                os.mkdir("weights/")
            url = 'https://drive.google.com/uc?id={}'.format(json_file['n2n-{}.pt'.format(self.noise)])
            gdown.download(url, "weights/n2n-{}.pt".format(self.noise), quiet=False)
    
    def load_model(self):   
        ckpt_dir = "weights/n2n-{}.pt".format(self.noise)
        self.model.load_state_dict(torch.load(ckpt_dir, self.map_location))

    def transform_img(self, img):
        w,h = img.size
        m = min(w,h)
        img = tvf.crop(img,0,0,m,m)
        img = tvf.resize(img,(320, 320))
        return img

    def inference(self,img):
        img = self.transform_img(img)
        img = torch.unsqueeze(tvf.to_tensor(img),dim=0)
        output = self.model(img).detach()
        dd = output.squeeze(0)
        denoised = tvf.to_pil_image(dd)
        denoised.show()
        return denoised

## Usage ##
# if __name__=="__main__":
#     img = Image.open('misc/testt.jpg')
#     mo = Noise2Noise(img)
#     mo.inference(img, noise='text')




