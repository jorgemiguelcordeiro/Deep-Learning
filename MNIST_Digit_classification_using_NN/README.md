# MNIST Handwritten Digit Classification using Deep Learning (Neural Network)

This project builds a deep learning model using Keras and TensorFlow to classify handwritten digits from the MNIST dataset.

## Table of Contents
1. Importing the Dependencies
2. Loading the MNIST Data
3. Encoding and Normalizing the Pixel Values
4. Building the Neural Network
5. Building a Predictive System

## 1. Importing the Dependencies
The required libraries such as TensorFlow, Keras, NumPy, and OpenCV are imported.

## 2. Loading the MNIST Data
The dataset is loaded from `keras.datasets.mnist`, which consists of 60,000 training images and 10,000 test images.

## 3. Encoding and Normalizing the Pixel Values
- The pixel values range from **0 to 255** and are normalized to the range **0 to 1** for better model performance.
- Labels are integer-encoded from 0-9.

## 4. Building the Neural Network
A simple feedforward neural network (FNN) is constructed using Keras Sequential API:
- **Flatten Layer**: Converts 28x28 images into a 1D array.
- **Hidden Layers**: Two Dense layers with 50 neurons and ReLU activation.
- **Output Layer**: 10 neurons (one for each digit), using Sigmoid activation.

## 5. Building a Predictive System
A function is created to take an input image, preprocess it, and predict the handwritten digit using the trained model.
<p align="center">
  <img src="https://github.com/user-attachments/assets/d510a6ce-ee1e-46db-b007-140a455bb249" width="50%" />
</p>


