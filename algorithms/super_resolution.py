# SRGAN
import torch
import os
import json
import gdown
from PIL import Image
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


class SRGan():
    def __init__(self):
        # self.img = img
        self.check_weights()
        self.srgan_generator = self.load_model()

    def check_weights(self):
        if os.path.exists("weights/checkpoint_srgan.pth.tar"):
            print("Found weights")
        else:
            print("Downloading weights")
            self.download_weights()

    def download_weights(self):
        with open("misc/weights_download.json") as fp:
            json_file = json.load(fp)
            if not os.path.exists("weights/"):
                os.mkdir("weights/")
            url = 'https://drive.google.com/uc?id={}'.format(json_file['checkpoint_srgan.pth.tar'])
            gdown.download(url, "weights/checkpoint_srgan.pth.tar", quiet=False)
    
    def load_model(self):   
        srgan_checkpoint = "weights/checkpoint_srgan.pth.tar"
        srgan_generator = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))['generator'].to(device)    
        srgan_generator.eval()
        return srgan_generator
    
    def convert_image(self, img, source, target):
        if source == 'pil':
            img = FT.to_tensor(img)

        elif source == '[0, 1]':
            pass  

        elif source == '[-1, 1]':
            img = (img + 1.) / 2.

        if target == 'pil':
            img = FT.to_pil_image(img)

        elif target == '[0, 255]':
            img = 255. * img

        elif target == '[0, 1]':
            pass 

        elif target == '[-1, 1]':
            img = 2. * img - 1.

        elif target == 'imagenet-norm':
            if img.ndimension() == 3:
                img = (img - imagenet_mean) / imagenet_std
            elif img.ndimension() == 4:
                img = (img - imagenet_mean_cuda) / imagenet_std_cuda
    
        return img

    def inference(self, img):

        img = img.convert('RGB')
        sr_img_srgan = self.srgan_generator(self.convert_image(img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
        sr_img_srgan = self.convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

        return sr_img_srgan


## Usage ##
# if __name__ == '__main__':
#     img = Image.open("sample.jpg")
#     obj = SRGan()
#     obj.inference(img)
