import streamlit as st

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

        st.info("Please upload your image in .jpg, .jpeg or .png")

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
    
    idx = 0

    while idx < len(imglist):
        
        for _ in range(len(imglist)):
            cols = st.beta_columns(2) 

            for col_num in range(2): 

                if idx <= len(imglist):
                    cols[col_num].image(imglist[idx], 
                        width=328, caption=captions[idx])
                    
                    idx+=1
                    