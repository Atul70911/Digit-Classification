*MNIST Digit Classification using CNN*
This repository contains a Convolutional Neural Network (CNN) implementation for classifying handwritten digits from the MNIST dataset. The model achieves high accuracy on the test set, demonstrating its effectiveness for digit recognition tasks.

Project Overview
The goal of this project is to create and train a CNN to classify handwritten digits from the MNIST dataset, which consists of 28x28 pixel grayscale images of digits (0-9).

Files
mnist_cnn.py: Python script containing the CNN model implementation, training, and evaluation.
README.md: This file.
Dependencies
To run this project, you'll need to have the following Python libraries installed:

numpy
pandas
tensorflow
keras
matplotlib
seaborn
scikit-learn
You can install these dependencies using pip:

bash
Copy code
pip install numpy pandas tensorflow keras matplotlib seaborn scikit-learn
Usage
Load the MNIST Dataset:

The dataset is automatically loaded and split into training and test sets using TensorFlow.

Preprocess the Data:

The images are reshaped and normalized to prepare them for input into the CNN.

Define the CNN Model:

The CNN architecture includes:

Two sets of convolutional layers, each followed by batch normalization, ReLU activation, max pooling, and dropout.
A flattening layer followed by two dense layers: one with ReLU activation and one with a softmax activation for classification.
Compile the Model:

The model is compiled using the Adam optimizer and categorical crossentropy loss function.

Train the Model:

The model is trained for 5 epochs with a batch size of 64.

Evaluate the Model:

The model's predictions are compared with the true labels to generate a confusion matrix, which is visualized using seaborn.

Results
The model achieves an accuracy of 99.27% on the MNIST test set. The confusion matrix provides insights into the classification performance.
