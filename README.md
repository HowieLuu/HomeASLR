# AlexaASLR
Training an ML model on American Sign Language Recognition (ASLR), then interfacing the model with Alexa to allow for sign language to speech 

## Current steps:
1. Gather data
    1. From various Kaggle sources
    2. From webcam data collection
2. Preprocess the data
    1. Images may differ from source to source, but the network must take in a pre-defined size
    2. Therefore we must either crop or resize input images
        1. [Rescale/crop images with TensorFlow](https://wandb.ai/ayush-thakur/dl-question-bank/reports/How-to-Handle-Images-of-Different-Sizes-in-a-Convolutional-Neural-Network--VmlldzoyMDk3NzQ)
        2. [Rescale/crop images with OpenCV](https://learnopencv.com/image-resizing-with-opencv/#resize-with-scaling-factor)
    3. We also must determine what size the input image should be
        1. [Small training resolution can improve performance](https://arxiv.org/pdf/1906.06423.pdf)
        2. [Image size in radiology deep learning](https://pubs.rsna.org/doi/full/10.1148/ryai.2019190015)
3. Building a model
    1. [TensorFlow implementation](https://towardsdatascience.com/sign-language-recognition-with-advanced-computer-vision-7b74f20f3442)
    2. [MediaPipe for webcam capture](https://www.youtube.com/watch?v=MJCSjXepaAM)
4. Alexa Interface
    1. [Node Red](https://nodered.org/)
    2. [Node Red Python Flow](https://flows.nodered.org/node/node-red-contrib-pythonshell)


## Possible data sources
1. [Mediapipe link has module to capture image streams from webcam](https://www.youtube.com/watch?v=MJCSjXepaAM)
2. [Kaggle ASL](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
    1. 87000 images
    2. 200x200 resolution
    3. Labels for space, delete, return
