import streamlit as st
import streamlit.components.v1 as components
# from annotated_text import annotated_text

from utils.function_call import *
from utils.deployment import *
from PIL import Image

opt = st.sidebar.selectbox("",("Home", "Resources", "Visualizer"))

#lrn = st.sidebar.button("Learn More")

#if lrn:
#        clf = st.checkbox("Classification")
#        if clf:
#            st.write("ABC")
#try:

if opt == "Home":

    #st.set_page_config(layout='wide')

    html_temp = '''
        <div>
        <h2></h2>
        <center><h3>Vision Dashboard - A One-Stop CV Learning Tool</h3></center>
        </div>

        <hr>
        Computer Vision (CV) is a growing field that attracts many beginners in the field of Machine Learning. According to research, visual information is mapped better in students’ minds and helps them retain information for a longer duration.
        However, the traditional educational methodology involves teaching theoretical concepts utilizing text-based explanations and audio. This results in most students not being able to visualize or understand the significant CV techniques, and thus students are unsure about how to approach CV as a field. 
        <br></br>
        This project, <b>VisionDash</b>, tries to eliminate the problem by giving live demos and an easy to use interface to study Computer Vision.
        Checkout the different sections in the sidebar: Resources will give you a theoretical base and our novel Visualizer will show you how the various techniques work instantly!
        <br></br>



        '''

    
    st.markdown(html_temp, unsafe_allow_html=True)

    st.header("")
    
    tree()
    st.header("")
    html_temp = '''
    
        <div style = "background-color: rgba(25,25,112,0.06); padding: 15px; padding-left: 15px; padding-right: 15px">
        <p>Computer vision works much the same as human vision, except humans have a head start. Human sight has the advantage of lifetimes of context to train how to tell objects apart, how far away they are, whether they are moving and whether there is something wrong in an image.</p>
        </div>
        '''

    
    st.markdown(html_temp, unsafe_allow_html=True)

    st.title("")

    carousel()
    

elif opt == "Resources":

    html_temp = '''
        <div>
        <center><h2>Resources</h2></center>
        </div>
        '''

    st.markdown(html_temp, unsafe_allow_html=True)
    opt2 = st.sidebar.selectbox("",("Image Classification", "Detection", "Segmentation", "Denoising", "Style Transfer", "Super Resolution"))

    if opt2 == "Image Classification" :

        tmp = """
        <h3>What is <span style="color:pink">Image Classification?</span></h3>
            Image Classification is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rules.
        
        <hr>
        
        <h3>How does it <span style="color:pink">Work?</span></h3>
            Image classification is a supervised learning problem: define a set of target classes (objects to identify in images), 
            and train a model to recognize them using labeled example photos. 
            Early computer vision models relied on raw pixel data as the input to the model. 
            However, raw pixel data alone doesn't provide a sufficiently stable representation to encompass the myriad variations of an object as captured in an image. 
            The position of the object, background behind the object, ambient lighting, camera angle, and camera focus all can produce fluctuation in raw pixel data; 
            these differences are significant enough that they cannot be corrected for by taking weighted averages of pixel RGB values.
        <br></br>
            
        """
        st.markdown(tmp, unsafe_allow_html=True)
        im = Image.open("misc/images/classification.png")
        st.image(im)
        tmp2 = """
            To see a live demo, go to our custom <b>Visualiser!</b> Upload any image and watch the magic as your computer tells you what that image contains!
        <hr>
        <h3>Would you like to <span style="color:pink">Know More?</span> Checkout these links!</h3>
        <ul>
            <li>
                <a href="https://developers.google.com/machine-learning/practica/image-classification">ML Practicum: Image Classification!</a>
            </li>
            <li>
                <a href="/visualizer"> Reference 2</a>
            </li>
        </ul>
        """

        st.markdown(tmp2, unsafe_allow_html=True)
        
    if opt2 == "Detection":

        tmp = """
        <h3>What is <span style="color:pink">Detection?</span></h3>
            Object Detection is a computer technology related to computer vision, image processing, and deep learning that deals with detecting instances of objects in images and videos.
        <hr>
        
        <h3>How does it <span style="color:pink">Work?</span></h3>
            
        <br></br>
        """
        st.markdown(tmp, unsafe_allow_html=True)
        im = Image.open("misc/images/face_detection.png")
        st.image(im)
        tmp2 = """
        
        <hr>
        <h3>Would you like to <span style="color:pink">Know More?</span> Checkout these links!</h3>
        <ul>
            <li>
                <a href="https://developers.google.com/machine-learning/practica/image-classification">ML Practicum: Image Classification!</a>
            </li>
            <li>
                <a href="/visualizer"> Reference 2</a>
            </li>
        </ul>

        """
        st.markdown(tmp2, unsafe_allow_html=True)


    if opt2 == "Segmentation":


        """
        Segmentation is the process of dividing an image into different regions based on the characteristics of pixels to identify objects or boundaries to simplify an image and more efficiently analyze it.

        """

    if opt2 == "Denoising":

        """
        Denoising refers to estimating the original image by suppressing noise from a noise-contaminated version of the image.

        """

    if opt2 == "Style Transfer":

        """
        Style transfer is a computer vision technique that takes two images—a content image and a style reference image—and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image.

        """

    if opt2 == "Super Resolution":

        """
        Super-resolution is based on the idea that a combination of low resolution (noisy) sequence of images of a scene can be used to generate a high resolution image or image sequence. Thus it attempts to reconstruct the original scene image with high resolution given a set of observed images at lower resolution.
        """

else:

    task = st.sidebar.selectbox("Select the Algorithm that you want to run!",("Classification", "Face Detection", "Object Detection", "Instance Segmentation", "Semantic Segmentation", "Denoising", "Style Transfer", "Super Resolution"))

    input_image = image_upload()
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
            
            style_image = Image.open("misc/images/style_images/{}.jpg".format(style))
            
            # Hacky way to centre image
            col1, col2, col3 = st.columns([4,10,4])
            
            with col1:
                st.write("")
            with col2:
                st.image(style_image, width=300)
            with col3:
                st.write("")

            style_image = style_image.resize((input_image.size))
            output = transfer(input_image, style)

            if st.button("Transfer Style"):
                display(input_image, captions=['Content Image', 'Style Image'], resimg=style_image)
                st.image(output, width=660, caption="Stylized Image!")

        elif task == "Super Resolution":
            output = superres(input_image)
            
            if st.button("Transform"):
                display(input_image, captions=['Uploaded Image', 'Image Transformed!'], resimg=output)

