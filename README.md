# Triplet-Loss
Fine tune and train a CNN pre-trained on Imagenet dataset, using Triplet Loss

# Implementation Details
The code is implemented using the Keras library with TensorFlow backend in a python environment.
The data_format is changed to 'channels_first' by default in the keras.json file.

# Files:
    init-
        --Start of the code.
        --Download and split dataset (Cifar 10 in the current example)
        --Fit model on the training and validation data.
        --Predict classes of the test data.
        
    fine_tune_model-
        --Contains two functions; fine_tuned_models and the wrapper function, data_generator.
        --The wrapper function is not used in the final run of the implementation.
       
    triplet_loss_functions-
        --Two different implementations of triplet loss.

# Background

Triplet loss was first implemented by Florian Schroff and Dmitry Kalenichenko in the paper that introduced FaceNet. It is computed using 
three images; 
a. anchor image     -   The user defined anchor.
b. positive image   -   Image of the same class as the anchor.
c. negative image   -   Image of a different class.

The premise of triplet loss is to separate the embeddings of a positive pair from a negative pair by a margin distance m / alpha.
The positive pair is the anchor and the positive image whereas the negative pair is the anchor and the negative image.

The mathematical fuction for triplet loss is as follows:

http://latex.codecogs.com/gif.latex?L%20%3D%20%5Csum_%7Bi%7D%5E%7BN%7D%20%5B%20%5Cleft%20%5C%7Cf%28x_%7Bi%7D%5E%7Ba%7D%29%20-%20f%28x_%7Bi%7D%5E%7Bp%7D%29%5Cright%20%5C%7C_%7B2%7D%5E%7B2%7D%20-%20%5Cleft%20%5C%7C%20f%28x_%7Bi%7D%5E%7Ba%7D%29%20-%20f%28x_%7Bi%7D%5E%7Bn%7D%29%20%5Cright%20%5C%7C_%7B2%7D%5E%7B2%7D%20&plus;%5Calpha%20%5D_%7B&plus;%7D

Triplet Loss can be implemented directly as a loss function in the compile method, or it can be implemented as a merge mode with the
anchor, positive and negative embeddings of three individual images as the three branches of the merge function.

# Problem
The code trains and fine-tunes a CNN model (ResNet50), pre-trained on the Imagenet dataset, by replacing the classifier of the CNN and 
using triplet loss. The First 15 layers of ResNet50 have been frozen to reduce the affect of overfitting to the new dataset. 

# Note
Dataset used is Cifar 10. However the images in Cifar 10 are of dimensions 32x32. However CNN models like ResNet50 / AlexNet / VGG16 /
GoogleNet etc. require image dimensions to be at least 197x197. 
Cifar 10 images can either be scaled up or padded neither of which are a good solution. 
It is better to use a dataset similar to Imagenet, with images of comparable size and information. 
The code contains Cifar 10 because it is already available with the Keras library. Add your own database in its place to run the code.
