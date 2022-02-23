import streamlit as st
import streamlit.components.v1 as components
from utils.function_call import *
from utils.deployment import *
from PIL import Image

set_bg_local("misc/images/bg.png")
opt = st.sidebar.selectbox("",("Home", "Resources", "Visualizer", "Frequently Asked Questions"))

if opt == "Home":

    html_temp = '''
        <div>
        <h2></h2>
        <center><h3>Vision Dashboard - A One-Stop CV Learning Tool</h3></center>
        </div>

        <hr>
        Computer Vision (CV) is a growing field that attracts many beginners in the field of Machine Learning. According to research, visual information is mapped better in students’ minds and helps them retain information for a longer duration.
        However, the traditional educational methodology involves teaching theoretical concepts utilizing text-based explanations and audio. This results in most students not being able to visualize or understand the significant CV techniques, and thus students are unsure about how to approach CV as a field. 
        <br></br>
        Computer vision works much the same as human vision, except humans have a head start. Human sight has the advantage of lifetimes of context to train how to tell objects apart, how far away they are, whether they are moving and whether there is something wrong in an image.
        <br></br>
        This project, <b>VisionDash</b>, tries to eliminate the problem by giving live demos and an easy to use interface to study Computer Vision.
        Checkout the different sections in the sidebar: Resources will give you a theoretical base and our novel Visualizer will show you how the various techniques work instantly!
        <br></br>



        '''

    
    st.markdown(html_temp, unsafe_allow_html=True)

    st.header("")
    
    tree()

    #st.title("")

    html_temp = """<hr>"""
    st.markdown(html_temp, unsafe_allow_html=True)

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
                <a href="https://www.v7labs.com/blog/image-classification-guide"> Image Classification Explained: An Introduction [+V7 Tutorial]</a>
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
            To make image recognition possible through machines, we need to train the algorithms that can learn and predict with accurate results. Let’s take an example – if you look at the image of a cat, you can easily tell it is a cat, but the image recognition algorithm works differently.
            Due to similar attributes, a machine can see it 75% cat, 10% dog, and 5% like other similar looks like an animal that are referred to as confidence score. And to predict the object accurately, the machine has to understand what exactly sees, then analyze comparing with the previous training to make the final prediction.
        <br></br>
        """
        st.markdown(tmp, unsafe_allow_html=True)
        im = Image.open("misc/images/face_detection.png")
        st.image(im)
        tmp2 = """
        To see a live demo, go to our custom <b>Visualiser!</b> Upload any image and watch the magic as your computer gives you bounding boxes around the objects it recognises in your image!
        <hr>
        <h3>Would you like to <span style="color:pink">Know More?</span> Checkout these links!</h3>
        <ul>
            <li>
                <a href="https://www.anolytics.ai/blog/what-is-ai-image-recognition-how-does-it-work/">Anolytics: Image Detection</a>
            </li>
            <li>
                <a href="https://paperswithcode.com/task/object-detection"> Papers related to Object Detection</a>
            </li>
        </ul>

        """
        st.markdown(tmp2, unsafe_allow_html=True)


    if opt2 == "Segmentation":

        tmp = """
        <h3>What is <span style="color:pink">Segmentation?</span></h3>
            Segmentation is the process of dividing an image into different regions based on the characteristics of pixels to identify objects or boundaries to simplify an image and more efficiently analyze it.
            <br></br>
            Segmentation can be divided into two essential types:
            <ol>
            <li>
            Instance Segmentation
            </li>
            <li>
            Semantic Segmentation
            </li>
            </ol>

        <hr>
        
        <h3>How does it <span style="color:pink">Work?</span></h3>
            Image segmentation creates a pixel-wise mask for each object in the image. This technique gives us a far more granular understanding of the object(s) in the image.
            This can be done through various techniques like:
            <ul>
            <li>
            Threshold Based Segmentation
            </li> 
            <li>
            Edge Based Segmentation
            </li> 
            <li>
            Region-Based Segmentation
            </li> 
            <li>
            Clustering Based Segmentation
            </li> 
            <li>
            Artificial Neural Network Based Segmentation
            </li> 
            </ul>
        <br></br>
        """
        st.markdown(tmp, unsafe_allow_html=True)
        im = Image.open("misc/images/semantic_segmentation.png")
        st.image(im, caption="Semantic Segmentation")
        im2 = Image.open("misc/images/instance_segmentation.png")
        st.image(im2, caption="Instance Segmentation")
        tmp2 = """
        To see a live demo, go to our custom <b>Visualiser!</b> Both types of segmentation are implemented, just upload an image and see the segmentation masks like the one shown above!
        <hr>
        <h3>Would you like to <span style="color:pink">Know More?</span> Checkout these links!</h3>
        <ul>
            <li>
            <a href="https://www.analytixlabs.co.in/blog/what-is-image-segmentation/">What is Image Segmentation?</a>
            </li>
            <li>
                <a href="https://towardsdatascience.com/image-segmentation-part-1-9f3db1ac1c50">Image Segmentation: Part 1</a>
            </li>
            <li>
                <a href="https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/#:~:text=By%20dividing%20the%20image%20into,is%20how%20image%20segmentation%20works.&text=Object%20detection%20builds%20a%20bounding,each%20class%20in%20the%20image.&text=Image%20segmentation%20creates%20a%20pixel,each%20object%20in%20the%20image."> Computer Vision Tutorial: A Step-by-Step Introduction to Image Segmentation Techniques (Part 1)</a>
            </li>
            <li>
            <a href="https://ai.stanford.edu/~syyeung/cvweb/tutorial3.html">Tutorial: Image Segmentation</a>
            </li>
        </ul>
        """
        st.markdown(tmp2, unsafe_allow_html=True)

    if opt2 == "Denoising":

        tmp = """
        <h3>What is <span style="color:pink">Denoising?</span></h3>
            Denoising refers to estimating the original image by suppressing noise from a noise-contaminated version of the image.
        <hr>
        
        <h3>How does it <span style="color:pink">Work?</span></h3>
        Mathematically, the problem of image denoising can be modeled as follows:
        𝑦=𝑥+𝑛
        where y is the observed noisy image, x is the unknown clean image, and n represents additive white Gaussian noise (AWGN) with standard deviation σ. The purpose of noise reduction is to decrease the noise in natural images while minimizing the loss of original features and improving the signal-to-noise ratio (SNR). This is done through various methods depending on the model used for denoising.
            
        <br></br>
        """
        st.markdown(tmp, unsafe_allow_html=True)
        im = Image.open("misc/images/denoising.png")
        st.image(im)
        tmp2 = """
        To see a live demo, go to our custom <b>Visualiser!</b> Our denoiser can clear two types of noises; gaussian and textual noise. See the magic in action!
        
        <hr>
        <h3>Would you like to <span style="color:pink">Know More?</span> Checkout these links!</h3>
        <ul>
        <li>
            <a href="https://uwaterloo.ca/vision-image-processing-lab/research-demos/image-denoising#:~:text=One%20of%20the%20fundamental%20challenges,contaminated%20version%20of%20the%20image">Image Denoising</a>
            </li>
            <li>
            <a href="https://analyticsindiamag.com/a-guide-to-different-types-of-noises-and-image-denoising-methods/">A Guide to Different Types of Noises and Image Denoising Methods</a>
            </li>
            <li>
                <a href="https://computergraphics.stackexchange.com/questions/6419/what-is-the-basic-idea-of-denoising">What is the basic idea of denoising?</a>
            </li>
            <li>
                <a href="https://arxiv.org/abs/1803.04189"> Noise2Noise Paper</a>
            </li>
        </ul>
        """
        st.markdown(tmp2, unsafe_allow_html=True)

    if opt2 == "Style Transfer":

        tmp = """
        <h3>What is <span style="color:pink">Style Transfer?</span></h3>
            Style transfer is a computer vision technique that takes two images—a content image and a style reference image—and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image.
        <hr>
        
        <h3>How does it <span style="color:pink">Work?</span></h3>
            NST employs a pre-trained Convolutional Neural Network with added loss functions to transfer style from one image to another and synthesize a newly generated image with the features we want to add.
            <br>
            Style transfer works by activating the neurons in a particular way, such that the output image and the content image should match particularly in the content, whereas the style image and the desired output image should match in texture, and capture the same style characteristics in the activation maps.
            <br></br>
            The required inputs to the model for image style transfer:
            <ul>
            <li>A Content Image – an image to which we want to transfer style to</li>
            <li>A Style Image – the style we want to transfer to the content image</li>
            <li>An Output Image (generated) – the final blend of content and style image</li>
            </ul>
        <br></br>
        """
        st.markdown(tmp, unsafe_allow_html=True)
        im = Image.open("misc/images/style_transfer.png")
        st.image(im)
        tmp2 = """
        To see a live demo, go to our custom <b>Visualiser!</b> There are four types of styles provided: Candy, Mosaic, Rain Princess, Udnie.
        Upload an image, choose one of the styles, and see your stylised image!
        <hr>
        <h3>Would you like to <span style="color:pink">Know More?</span> Checkout these links!</h3>
        <ul>
            <li>
                <a href="https://arxiv.org/abs/1508.06576">Style Transfer Paper</a>
            </li>
            <li>
                <a href="https://www.v7labs.com/blog/neural-style-transfer#:~:text=Style%20transfer%20works%20by%20activating,characteristics%20in%20the%20activation%20maps."> Neural Style Transfer: Everything You Need to Know [Guide]</a>
            </li>
        </ul>
        """
        st.markdown(tmp2, unsafe_allow_html=True)

    if opt2 == "Super Resolution":

        tmp = """
        <h3>What is <span style="color:pink">Super Resolution?</span></h3>
            Super-resolution is based on the idea that a combination of low resolution (noisy) sequence of images of a scene can be used to generate a high resolution image or image sequence. Thus it attempts to reconstruct the original scene image with high resolution given a set of observed images at lower resolution.
        <hr>
        
        <h3>How does it <span style="color:pink">Work?</span></h3>
            Low resolution images can be modeled from high resolution images using the below formula, where D is the degradation function, Iy is the high resolution image, Ix is the low resolution image, and σ is the noise.
            <br>
            Ix = D(Iy;σ)
            <br>
            The degradation parameters D and σ are unknown; only the high resolution image and the corresponding low resolution image are provided. The task of the neural network is to find the inverse function of degradation using just the HR and LR image data.
            There are many methods used to solve this task. We will cover the following:
            <ul>
            <li>Pre-Upsampling Super Resolution</li>
            <li>Post-Upsampling Super Resolution</li>
            <li>Residual Networks</li>
            <li>Multi-Stage Residual Networks</li>
            <li>Recursive Networks</li>
            <li>Progressive Reconstruction Networks</li>
            <li>Multi-Branch Networks</li>
            <li>Attention-Based Networks</li>
            <li>Generative Models</li>
            </ul>
        <br></br>
        """
        st.markdown(tmp, unsafe_allow_html=True)
        im = Image.open("misc/images/superresolution.png")
        st.image(im)
        tmp2 = """
        To see a live demo, go to our custom <b>Visualiser!</b> Checkout how it quadruples the resolution of your image!
        <hr>
        <h3>Would you like to <span style="color:pink">Know More?</span> Checkout these links!</h3>
        <ul>
            <li>
                <a href="https://blog.paperspace.com/image-super-resolution/">Image Super-Resolution: A Comprehensive Review</a>
            </li>
            <li>
                <a href="https://arxiv.org/abs/1609.04802"> Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network</a>
            </li>
        </ul>
        """
        st.markdown(tmp2, unsafe_allow_html=True)

elif opt == "Frequently Asked Questions":
    st.write("Hi")

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

