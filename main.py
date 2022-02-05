import streamlit as st
import cv2
import urllib.request
import time

from utils.function_call import *
from utils.deployment import *

task = st.sidebar.selectbox("", ("Home", "Classification", "Face Detection", "Object Detection", "Instance Segmentation", "Semantic Segmentation"))

try:

    if task == "Home":

        html_temp = '''
            <div>
            <h2></h2>
            <center><h1>Vision Dashboard</h1></center>
            </div>
            '''

        st.markdown(html_temp, unsafe_allow_html=True)

    else:

        input_image = image_upload()
        #st.image(input_image, width = 300, caption = 'Uploaded Image')

        if task == "Classification":

            output = classify(input_image)

            if st.button("Classify"):
                display(input_image, captions=['Uploaded Image'])
                st.success("The Uploaded Image has been Classified as '{}'".format(output))
                
        elif task == "Face Detection":
            
            output = face_detect(input_image)

            if st.button("Detect Face"):
                display(input_image, captions=['Uploaded Image', 'Face(s) Detected!'], resimg=output)

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
except:

    pass