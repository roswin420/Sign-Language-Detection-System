# Real-Time Gesture Detection Using 3D DCNN-LSTM Model

This repository contains the implementation of a real-time hand gesture recognition system using a 3D DCNN-LSTM model. The system recognizes gestures in real-time using video input from a webcam and classifies them into different categories.

## Table of Contents

* Introduction
* Tech Stack
* System Architecture
* CNN LSTM Model Implementation
* Installation
* Usage
* Model Testing
* Model Evaluation
* Results
* Conclusion

## Introduction

Hand gesture recognition systems have gained significant attention due to their wide range of applications, including virtual prototyping, sign language analysis, and medical training. This project presents a novel real-time method for hand gesture recognition using a 3D DCNN-LSTM model. The main components of the system include video pre-processing, feature extraction, training the DCNN-LSTM model, and validating/testing the trained model.

## Tech Stack
* ML Model:  3D Convolutional Neural Networks +Long Short Term Memory (3D CNN-LSTM)
* Prog Lang:  Python 
* Libraries: OpenCV, Pandas, Keras, Tensorflow, Pickle
* WebApp: Flask Framework
* Development Env: Google Collab Cloud

## System Architecture

The system architecture involves capturing video frames from the webcam, converting them into a set of images, and processing them using the trained DCNN-LSTM model.

![image](https://user-images.githubusercontent.com/44722717/232201223-e6758155-2e79-450b-9cbf-d3ed3a6c5559.png)

Here's a brief explanation of the architecture:
1. Webcam: Captures the user's gestures in real-time as a video stream.
2. Gesture recognition system: Receives the video stream and extracts individual frames (images) from it.
3. Hand detection and feature extraction: The system identifies pixels in the images where only the hand is present, then extracts various features from these images using a deep learning model called the DCNN-LSTM Network.
4. Feature matching: The extracted features are matched with various gesture classes in the trained CNN model.
5. Gesture recognition output: The matching class name, or label of the recognized gesture, is provided as the result. The user can then read the text form of the gesture on the system screen.

This architecture allows for efficient and accurate gesture recognition in real-time by using deep learning techniques to process and classify the captured images.

##  CNN LSTM Model Implementation

![image](https://user-images.githubusercontent.com/44722717/232201297-d63ad6aa-5b25-4e77-bcf7-7a09d2bb0742.png)

3D DCNN-LSTM model used for real-time gesture detection. Here is a brief overview of the process:

1. Video preprocessing: The gesture dataset consists of 20 different gestures, with around 1000 videos per gesture. Each video is split into 15 frames of size 64x64, followed by binarization, thresholding, resizing, and normalization operations before being fed into the DCNN-LSTM model.
2. Feature extraction: The main goal of feature extraction is to obtain the most relevant information from the original data and represent it in a lower-dimensional space. The input video of shape 15x64x64 is passed through two convolution layers with 3D kernels, followed by ReLU activation and max-pooling to reduce dimensionality. The output from the second convolution layer is passed to the Conv-LSTM layer, which gives 5760 outputs of shape 12x12x40.
3. Fully connected layers and output: The output from the Conv-LSTM layer is passed to two fully connected layers, and the final output is obtained by passing the result through a softmax layer with four nodes, representing four different classes of gestures.
4. System training: The 3D DCNN-LSTM model is trained to recognize four different gestures using a dataset from the Jester-20bln Dataset. The training process consists of a forward phase, where the input is passed through the network, and a backward phase, where gradients are backpropagated, and weights are updated. The loss function used is Sparse Categorical Cross-entropy.
5. System testing:The model is tested using various videos that are preprocessed and feature-extracted before being given as input. The output obtained from the model is checked against the actual output, and accuracy is computed based on the deviation of outputs concerning expected outputs.

##  Installation

1. Clone the repository
```git clone https://github.com/your_username/Real-Time-Gesture-Detection-3D-DCNN-LSTM.git```
2. Install the required dependencies
```pip install -r requirements.txt```
3. Download the pre-trained model and dataset (jester-20bln) from the provided link and place them in the appropriate directories.

## Usage
1. Run the main script to start the real-time gesture detection system:
```python main.py```
A window will open, displaying the video feed from the webcam. Perform hand gestures in front of the camera, and the system will recognize and display the corresponding label in the top-left corner of the window.

## Model Testing
![image](https://user-images.githubusercontent.com/44722717/232201101-3acd80d9-87e1-41e4-8692-fbedb3706e85.png)
![image](https://user-images.githubusercontent.com/44722717/232201139-0d45f090-5dc1-4476-9c4e-a3e4d3579160.png)
![image](https://user-images.githubusercontent.com/44722717/232201155-82715c68-75ab-4537-928c-10b4e3de7b5c.png)
![image](https://user-images.githubusercontent.com/44722717/232201169-84c5b0ee-0606-4aac-8646-5f4540509398.png)

## Model Evaluation

Using the open-cv2 library, the real-time input is captured and fed to the 3D DCNN-LSTM model. The model can predict gestures accurately in various conditions, with 95.3% accuracy for training data and 90.1% for test data. With increasing epochs, accuracy improves, but overfitting may occur after two epochs. The loss function, Sparse Categorical Cross-entropy, shows the model efficiently learns to extract features in a smaller number of epochs.

![image](https://user-images.githubusercontent.com/44722717/232201013-0f570807-fe2a-49c0-b2fe-78ec499e8e51.png)

![image](https://user-images.githubusercontent.com/44722717/232201044-95a265de-405b-48d6-a32b-68151aaa3c4c.png)



## Results
The 3D DCNN-LSTM model achieved an accuracy of 95.3% on the training data and 90.1% on the test data. The model was tested in different lighting conditions and with gestures performed by different people, demonstrating robustness and high accuracy.

## Conclusion
The real-time hand gesture recognition system using a 3D DCNN-LSTM model achieved a high accuracy of 90.1%. The system is based on the Google TensorFlow 2.0 deep learning framework and is hosted on Google Cloud. This approach provides faster and more reliable results compared to offline machine learning libraries. It can further be used for the development of mobile applications without requiring significant on-device computation, reducing the load on local devices.
