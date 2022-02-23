# VisionDash - A One-Stop CV Learning Tool
Computer Vision (CV) is a growing field that attracts many beginners in the field of Machine Learning. According to research, visual information is mapped better in students’ minds and helps them retain information for a longer duration. However, the traditional educational methodology involves teaching theoretical concepts utilizing text-based explanations and audio. This results in most students not being able to visualize or understand the significant CV techniques, and thus students are unsure about how to approach CV as a field. In addition, CV models tend to be computationally heavy, expensive, and difficult to run from a beginner’s point of view, which discourages students from pursuing the field seriously. This paper presents a method of demonstrating CV algorithms using a Vision Dashboard, keeping the aforementioned issues in mind. 

Our approach allows students to run various CV methods on any image compiled on a single dashboard. This helps students visualize techniques like Object Detection, Instance Segmentation, Semantic Segmentation, Style Transfer, Image Classification, Super Resolution, Denoising, Image generation using GANs, and Face Detection efficiently, serving as an effective teaching tool.

## Implementation Details
Our project, VisionDash, consists of a dashboard providing the users with an option to overview various CV tasks, learn about different CV techniques utilizing the resources provided on the dashboard and supplement their knowledge with the SOTA implementations of each algorithm. 

The algorithms provided are divided into broad categories: Image Classification, Detection, Segmentation, Denoising, Generative Adversarial Networks, and Style Transfer. These categories are further divided into the specific tasks of Image Classification, Object Detection and Face Detection, Instance Segmentation and Semantic Segmentation, Noise2Noise, Super Resolution GAN, and Fast Style Transfer. 

Task | Model | Source |
:----------: | :-----------: | :-----------: |
Image Classifiation | EfficientNetb7 | [Torchvision](https://pytorch.org/vision/stable/models.html)
Face Detection | MTCNN | [Facenet-Pytorch](https://github.com/timesler/facenet-pytorch)
Object Detection | RetinaNet | [Torchvision](https://pytorch.org/vision/stable/models.html)
Super Resolution | SRGANs | [Open-source](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution) 
Denoising | Noise2Noise | [Open Source](https://github.com/joeylitalien/noise2noise-pytorch)
Semantic Segmentation | DeepLabV3 | [Torchvision](https://pytorch.org/vision/stable/models.html)
Instance Segmentation | Mask RCNN | [Torchvision](https://pytorch.org/vision/stable/models.html)
Style Transfer | Fast Style Transfer | [Open Souce](https://github.com/pytorch/examples)


These models have been integrated into a single dashboard built using Streamlit which turns data scripts into shareable web apps. Different widgets have been used to make the user interface as interactive and visually impactful as possible. 
As a part of VisionDash, we provide resources to serve as a self-study tool for each CV technique implemented. This knowledge can be further augmented by the FAQs provided for each section.

## Usage
This app is deployed on Streamlit. Check out the demo at [https://share.streamlit.io/sashrika15/visiondash/main/main.py](https://share.streamlit.io/sashrika15/visiondash/main/main.py) 

## Contributors
- [Sashrika Surya](https://github.com/sashrika15)
- [Srijarko Roy](https://github.com/srijarkoroy)
