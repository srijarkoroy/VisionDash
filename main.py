from turtle import width
import streamlit as st
import streamlit.components.v1 as components

import cv2
import urllib.request
import time

from utils.function_call import *
from utils.deployment import *

task = st.sidebar.selectbox("Select the Algorithm that you want to run!", ("Home", "Classification", "Face Detection", "Object Detection", "Instance Segmentation", "Semantic Segmentation", "Denoising", "Style Transfer"))

#try:

if task == "Home":

    #st.set_page_config(layout='wide')

    html_temp = '''
        <div>
        <h2></h2>
        <center><h3>Vision Dashboard - A One-Stop CV Learning Tool</h3></center>
        </div>
        '''

    st.markdown(html_temp, unsafe_allow_html=True)
    
    tree()

    carousel()

else:

    input_image = image_upload()
    #st.image(input_image, width = 300, caption = 'Uploaded Image')

    if input_image is not None:
        if task == "Classification":

            output = classify(input_image)

            if st.button("Classify"):
                display(input_image, captions=['Uploaded Image'])
                st.success("The Uploaded Image has been Classified as '{}'".format(output))
                
        elif task == "Face Detection":
            
            try:
                output = face_detect(input_image)

                if st.button("Detect Face"):
                    display(input_image, captions=['Uploaded Image', 'Face(s) Detected!'], resimg=output)

            except:
                st.info("No Face(s) present in Uploaded Image! Please upload an Image having human face(s) to detect.")

        elif task == "Object Detection":
            
            output = object_detect(input_image)

            if st.button("Detect Object"):
                display(input_image, captions=['Uploaded Image', 'Object(s) Detected!'], resimg=output)

        elif task == "Instance Segmentation":

            output = instance_segment(input_image)

            if st.button("Segment"):
                display(input_image, captions=['Uploaded Image', 'Image Segmented!'], resimg=output)

        elif task == "Semantic Segmentation":

            output = semantic_segment(input_image)

            if st.button("Segment"):
                display(input_image, captions=['Uploaded Image', 'Image Segmented!'], resimg=output)

        elif task == "Denoising":
            
            noise = st.radio("Please Select a Noise type",("text","gaussian"))
            output = denoise(input_image, noise)

            if st.button("Denoise"):
                display(input_image, captions=['Uploaded Image', 'Image Denoised!'], resimg=output)

        elif task == "Style Transfer":

            style = st.radio("Please Select a Style Image", ("candy", "mosaic", "rain_princess", "udnie"))
            
            if style == "candy":
                img = Image.open("misc/images/style_images/candy.jpg")
            elif style == "mosiac":
                img = Image.open("misc/images/style_images/mosiac.jpg")
            elif style == "rain_princess":
                img = Image.open("misc/images/style_images/rain_princess.jpg")
            else:
                img = Image.open("misc/images/style_images/udnie.jpg")
            
            # Hacky way to centre image
            col1, col2, col3 = st.columns([4,10,4])
            with col1:
                st.write("")
            with col2:
                st.image(img, width=300)
            with col3:
                st.write("")

            output = transfer(input_image, style)
            if st.button("Transfer Style"):
                display(input_image, captions=['Uploaded Image', 'Styled Image!'], resimg=output)
