##
## --------------------------------------------------------
##  Nebojsa Djosic  
##  CP8318 Machine Learning - Assignment 2
##  2024-10-25
##  Copyright 2024 Nebojsa Djosic
## --------------------------------------------------------
import logging
import os
from pathlib import Path
import subprocess
import sys

INSTALL_DEPS = False
DATA_DIR = 'data'
RESULTS_DIR = 'results'
PWD = '.'
## configure logging, install dependencies...
if __name__ == '__main__':  
    PWD = Path(__file__).resolve().parent
    DATA_DIR = PWD / 'data'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = PWD / 'results'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(RESULTS_DIR / "script_Nebojsa_Djosic.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ])

    ## suppress unnecessary logging...
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)

    if INSTALL_DEPS:
        packages = [ ## List of dependencies to install
            "tensorflow",
            "matplotlib",
            "pandas",
            "scikit-learn",
            "ucimlrepo"
        ]
        for package in packages:
            logging.info(f'Installing dependencies: {package}')
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


msg = """
Assignment 3: Neural networks
"""
logging.info(msg)

import matplotlib.pyplot as plt
# import tensorflow as tf
from tensorflow import keras
import numpy as np

msg = """
PART 1: Implementing a neural network using Keras
The goal is to get introduced to Keras. 
Start by installing TensorFlow.
Keras is a high-level Deep Learning API that allows you to easily build, train, evaluate and execute all sorts of neural network
"""
logging.info(msg)

msg = 'From Keras, load the MNIST digits classification dataset'
logging.info(msg)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() #5 points

msg = 'Visualize the first 10 instances (digits) from the dataset'
plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray') #5 points
    plt.axis('off')  
plt.show()


msg = 'Verify the shape of the instances and associated label'
logging.info(msg)
logging.debug(f'x_train.shape: {x_train.shape}') ## (60000, 28, 28)
logging.info(f'In the training set, there are {x_train.shape[0]}, instances (2D grayscale image data with 28×28 pixels. \
In turn, every image is represented as a 28×28 array rather than a 1D array of size 784. \
Pixel values range from 0 (white) to 255 (black).) \
The associated labels are digits ranging from 0 to 9.') #5 points

msg = 'Scale the input feature down to 0-1 values, by dividing them by 255.0'
logging.info(msg)
x_train = x_train.astype('float32') / 255.0 #5 points
x_test = x_test.astype('float32') / 255.0 #5 points

msg = """
Create a Sequential model. A sequential model is a stack of layers connected sequentially.
This is the simplest kind of model for neural networks in Keras.
"""
logging.info(msg)
model = keras.Sequential() #5 points

msg = """
Build a first layer to the model, that will convert each 2D image into a 1D array. 
# For this, add a 'Flatten layer', and specify the shape (input_shape) of the instances [28,28]. 
"""
logging.info(msg)
model.add(keras.layers.Flatten(input_shape=[28, 28])) #5 points

msg = """
Build the first hidden layer to the model. 
For this, use a 'Dense layer' with 300 neurons, and use the ReLU activation function. 
A dense Layer is simple layer of neurons in which each neuron receives input from all the neurons of previous layer.
"""
logging.info(msg)
model.add(keras.layers.Dense(300, activation='relu')) #5 points

msg = """
Build a second hidden layer to the model. 
For this, use a 'Dense layer' with 100 neurons, also using the ReLU activation function.
"""
logging.info(msg)
model.add(keras.layers.Dense(100, activation='relu')) #5 points

msg = """
Build an output layer to the model.
For this, use a 'Dense layer' with 10 neurons (one per class), using the softmax activation function.
"""
logging.info(msg)
model.add(keras.layers.Dense(10, activation='softmax')) #5 points

logging.info('Explain why the softmax activation function was used for the output layer?')
logging.info("The softmax action function was used for the output layer because this is a multi-class classificatio, the output must be classified in 10 classes (categories), representing digits 0 - 9.") #5 points

logging.info("Use the model’s summary() method to display the model’s layers. Then complete the following blanks.") 
# (there is no need to write anycode to retrieve the information, you can simply type-in your answers directly)
model.summary()
logging.info("The size of the first hidden layer is 300. None means the batch size \
              is not fixed and can be any number. The total number of parameters \
             of the first hidden layer is 235,500, which refers to the total number of weights and biases.\
             since this is a fully connected layer (dense layer), each neuron in this, first, hidden layer \
             is connected to all 784 neurons in the input layer. Therefore, the total number of weights \
             is 784 * 300 = 235,200. To this number we add one bias parameter per neuron in this layer, \
             which is 300. Therefore, the total number of parameters is the sum of weights and biases or \
             235,200 + 300 = 235,500.") #8 points

msg = """
Call the method compile() on your model to specify the loss function and the optimizer to use. 
Set the loss function to be "sparse_categorical_crossentropy" and use the stochastic gradient descent optimizer.
"""
logging.info(msg)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd') #8 points

logging.info("Research then explain what is an epoch in machine learning.") #6 points
logging.info("""
    An epoch is one of hyperparameters. A hyperparameter value is set by the developer and is a part of
    configuration or code. This means that the values of hyperparameters are not learned, not changed during
    the training. Hyperparameters are typically used to control the learning process and/or the behavior 
    during the training depending on the machine learning algorithm. They are different from model parameters.
    Model parameters which are learned during training from the data. 
    The epoch hyperparameter defines the number of times that the learning algorithm will process (loop) through 
    the entire training dataset. One epoch means that each sample, instance (row) in the training dataset 
    has been used to update the internal model parameters. An epoch can have one or more batches. 
    For example, if we have 1,000 images and our batch size is 500, it means it will take 2 iterations to complete 1 epoch. 
""") #6 points

msg = """
Training the model: call the method fit(). As usual, you should pass the input features (x_train) 
and the associated target classes (y_train). This time, also set the number of epochs to 20.
"""
logging.info(msg)
model.fit(x_train, y_train, epochs=20) #6 points

logging.info('Test the model: use the method predict() to predict the labels of the first 10 instances of the test set')
plt.close('all')
y_pred = model.predict(x_test[:10]) #6 points
plt.figure(figsize=(10,4)) ## fix: labels were over images
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.title('Predicted label: ' + str(np.argmax(y_pred[i])), fontsize=10, pad=10) #6 points  ## fix: labels were over images
    plt.imshow(x_test[i], cmap='gray') #6 points
    plt.axis('off')
plt.tight_layout()  ## fix: labels were over images
plt.subplots_adjust(top=0.85)  ## fix: labels were over images
plt.show()


