import streamlit as st
import streamlit.components.v1 as components
import graphviz as graphviz

import torch
import torchvision
from torchvision import transforms as T

from PIL import Image
import urllib.request
import time
import itertools

def image_upload():

    html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Please upload Image to run the Algorithm</h3></center>
    </div>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    try:

        opt = st.selectbox("How do you want to upload the image?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
        
        if opt == 'Upload image from device':
            
            file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
            st.set_option('deprecation.showfileUploaderEncoding', False)
            if file is not None:
                input_image = Image.open(file)

        elif opt == 'Upload image via link':
            
            try:
                img = st.text_input('Enter the Image Address')
                input_image = Image.open(urllib.request.urlopen(img))
            
            except:
                if st.button('Submit'):
                    show = st.error("Please Enter a valid Image Address!")
                    time.sleep(4)
                    show.empty()

        return input_image

    except:

        st.info("Please upload your image in '.jpg', '.jpeg' or '.png'")

def display(input_image, captions, resimg=None):

    if resimg == None:
        st.image(input_image, width = 300, caption = 'Uploaded Image')

    else:

        display_trans = T.Compose([T.Resize(256)])
        orgimg = display_trans(input_image)
        images = [orgimg, resimg]

        #captions=['Uploaded Image', 'Image after Segmentation']

        #image_iter = paginator("", images)
        #index, fimg = map(list, zip(*image_iter))
        #st.image(fimg, width = 328, caption = captions)

        columns(images, captions)
      

def paginator(label, items, items_per_page=2):

    max_index =  items_per_page
    return itertools.islice(enumerate(items), max_index)

def columns(imglist, captions):
    
    try:
        idx = 0

        while idx < len(imglist):
            
            for _ in range(len(imglist)):
                cols = st.columns(2) 

                for col_num in range(2): 

                    if idx <= len(imglist):
                        cols[col_num].image(imglist[idx], 
                            width=328, caption=captions[idx])
                        
                        idx+=1
                        
    except:

        pass


def carousel():

    imageCarouselComponent = components.declare_component("image-carousel-component", path="misc/frontend/public")

    imageUrls = ["https://user-images.githubusercontent.com/66861243/154862761-5a7fd3ca-8217-42ba-a148-4e2ace15e4de.png",
        "https://user-images.githubusercontent.com/66861243/154862787-abe274cc-edb5-4b52-bc60-a6a1dc3b4039.png",
        "https://user-images.githubusercontent.com/66861243/154862810-88f95368-2719-4223-810e-987adeaf0c58.png",
        "https://user-images.githubusercontent.com/66861243/154862825-1a4cc32b-cd76-422f-9df1-466e47b9381a.png",
        "https://user-images.githubusercontent.com/66861243/154862827-457d680e-d7f6-4876-a6dd-5984813a6ccc.png",
        "https://user-images.githubusercontent.com/66861243/154862832-3030bee2-3229-40bf-a489-cdc88648c9a4.png",
        "https://user-images.githubusercontent.com/66861243/154862833-bf1d2888-dd0e-4177-8326-5d3481aba4dd.png"
        ]

    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

    if selectedImageUrl is not None:
        st.image(selectedImageUrl)

def tree():

    st.graphviz_chart('''
    digraph {
        node [fontsize = 9.5];
        CV_Task -> Image_Classification
        CV_Task -> Detection
        Detection -> Face_Detection
        Detection -> Object_Detection
        CV_Task -> GANs
        GANs -> SRGANs
        CV_Task -> Denoising
        CV_Task -> Segmentation
        Segmentation -> Instance_Segmentation
        Segmentation -> Semantic_Segmentation
        CV_Task -> Style_Transfer
    }
    ''')
