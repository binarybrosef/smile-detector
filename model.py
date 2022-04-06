##--Binary classification model classifying human faces in images as
##--similing or not smiling. A computationally low-cost, architecturally simple
##--convolutional neural network consisting of a single convolutional layer is used 
##--for binary classification. The model is trained on the "Happy House" dataset available
##--from Coursera's Deep Learning Specialization online course, and is further trained on
##--a supplemental dataset available from https://github.com/hromi/SMILEsmileD. 
##--The model can be evaluated on 64x64 RGB images. inference.py provides a method for resizing
##--user-provided images.

##--model.py constructs, trains, and evaluates a binary classification model with a 2D conv
##--layer having 32 filters, a kernel size of 7, and a stride of 1.


import matplotlib.pyplot as plt             
import numpy as np
import h5py
import tensorflow as tf
import tensorflow.keras.layers as tfl
import os
from PIL import Image


#Get training/test sets, labels, classes
def load_dataset():

    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #Training set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #Training set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #Test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) #Test set labels

    classes = np.array(test_dataset["list_classes"][:]) #Class labels

    #Convert rank-1 arrays into row vectors    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#Get supplemental training data from SMILEs dataset (https://github.com/hromi/SMILEsmileD)
def load_suppdata():

    #Set file path to negative and positive examples
    PATHS = {
                'neg': os.getcwd() + '\\SMILEs\\negatives\\negatives7',
                'pos': os.getcwd() + '\\SMILEs\\positives\\positives7'
            }

    #Build respective lists of all files that comprise positive and negative sets
    img_lists = {
                    'neg': os.listdir(PATHS['neg']),
                    'pos': os.listdir(PATHS['pos'])
                }

    labels = {'neg': 0, 'pos': 1}

    length = len(img_lists['neg']) + len(img_lists['pos'])
    images_array = np.empty((length,64,64,3))
    supp_y = np.empty((length,1))

    #For each positive and negative example, get numpy representation, normalize,
    #and place in 4D matrix with batch_size axis (axis=0).
    i = 0
    for key in img_lists.keys():
        for image in img_lists[key]:
            img = Image.open(PATHS[key] + '\\' + image)
            img_array = np.array(img)
            img_array = img_array/255.

            #Ssupplemental images are greyscale and of shape (64,64). To join them to the
            #original training data (of shape (64,64,3)), add another axis via expand_dims(). 
            #Use swapaxes() to move the new axis to the end, resulting in a shape of (64,64,1). 
            img_array = np.expand_dims(img_array, axis=1)
            img_array = np.swapaxes(img_array, 1, 2)

            #Place array version of image in array with batch size axis so it can be supplied to model.
            #The single greyscale color channel axis is broadcast to the other color channels in
            #images_array.
            images_array[i] = img_array

            #Set label for given image in supp_y vector 
            supp_y[i] = labels[key]

            i += 1

    return images_array, supp_y
    
#Create model
def Model(num_filters, kernel_size, stride):
    
    model = tf.keras.Sequential([
        
            tfl.ZeroPadding2D(3, input_shape=(64, 64, 3)),
            tfl.Conv2D(num_filters, kernel_size, stride),
            tfl.BatchNormalization(axis=3),
            tfl.ReLU(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(1, activation="sigmoid"),
            
            ])
    
    return model

#Show single image in dataset at index
def show_image(dataset, index):
    plt.imshow(dataset[index])
    plt.show()


np.random.seed(1)

#Load training set, test set, classes
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#Normalize images
X_train = X_train_orig/255.
X_test = X_test_orig/255.

#Append supplemental images, which are normalized in load_suppdata()
X_supp, Y_supp = load_suppdata()
X_train = np.append(X_train, X_supp, axis=0)

#Reshape label vectors
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

#Append supplemental labels
Y_train = np.append(Y_train, Y_supp, axis=0)

#Build model
model = Model(32, 7, 1)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Train model
model.fit(X_train, Y_train, epochs=10, batch_size=16)

#Save model
model.save(os.getcwd() + '\\model')

#Evaluate model against test set 
model.evaluate(X_test, Y_test)

