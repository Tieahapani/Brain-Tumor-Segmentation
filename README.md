# Brain Tumor Classification using Deep Learning
A convolutional neural network (CNN) based model for brain tumor classification from MRI scans. This repository contains the implementation of a PyTorch-based deep learning model that can accurately classify brain MRI images as either containing tumors ("yes") or not containing tumors ("no").
Overview

This project implements a CNN architecture for binary classification of brain MRI images. The model is designed to determine whether a given brain MRI scan contains a tumor, which can assist medical professionals in diagnosis and screening.
Dataset
The model was trained on a brain tumor dataset containing:

"yes" class: Images with brain tumors (~155 images)
"no" class: Images without brain tumors (~98 images)

The dataset has the following structure:
Copybrain_tumor_dataset/
├── yes/  # Images with tumors
│   └── [Y1.jpg, Y2.jpg, Y3.jpg, ...]
├── no/   # Images without tumors
│   └── [N1.jpg, N2.jpg, N3.jpg, ...]

Technologies Used : 

PyTorch: Main deep learning framework
OpenCV (cv2): For image processing and augmentation
scikit-learn: For evaluation metrics and train/test splitting
NumPy: For numerical operations
Matplotlib: For visualization
CUDA: For GPU acceleration on Nvidia hardware

Requirements : 

Python 3.x
PyTorch
OpenCV-Python
NumPy
Matplotlib
scikit-learn
CUDA and cuDNN (for GPU acceleration)
Nvidia GPU (for faster training)

Installation
bashCopy# Clone the repository
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification

# Install required packages
pip install torch torchvision
pip install opencv-python
pip install scikit-learn numpy matplotlib
Data Preprocessing
The implementation includes:

Image resizing to a standard dimension
Normalization
Data augmentation techniques (possible transformations include rotation, flipping, etc.)
Train/validation/test splitting

Model Architecture
The model is a Convolutional Neural Network with:

Multiple convolutional layers with ReLU activations
MaxPooling layers for downsampling
Batch normalization for faster and more stable training
Dropout for regularization
Fully connected layers for final classification

Training
bashCopy# To train the model
python train.py
The model is trained using:

Binary Cross-Entropy Loss
Adam optimizer
GPU acceleration for faster training



High accuracy in distinguishing between MRI scans with and without tumors
Fast inference time, suitable for clinical applications

Visualization
The repository includes code for visualizing:

Training/validation loss and accuracy curves
Sample predictions on test images
Confusion matrix for model evaluation

Usage in Google Colab
This project was developed and can be run in Google Colab. The repository includes Colab notebooks for easy reproduction of results.
Future Work

Implementation of more advanced architectures
Extension to multi-class tumor classification
Integration with web or mobile interfaces for easier access


The brain tumor dataset used for training and evaluation
PyTorch and CUDA for enabling efficient deep learning implementation
