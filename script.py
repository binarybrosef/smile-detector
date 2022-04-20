# Script comprising functions for transfer learning using a MobileNetv2 (MNv2) pre-trained on imagenet.
# Functions are provided for training the MNv2 to perform binary classification on 64x64 greyscale 
# face-containing images as smiling/not-smiling. custom_model() outputs MNv2 with untrained dense output layer
# that takes augmented face-containing images, produced via a sequential data augmentation model output from
# augment_data(), as input, where data augmentation includes flipping and rotating images.

# train_frozen_model() trains MNv2 with only the untrained dense output layer unfrozen, and with all other layers frozen.
# train_unfrozen_model() trains MNv2 with all layers, starting with the layer identified by the limit keyword argument, 
# unfrozen. Default value for limit is 118, meaning by default training will occur with layer 118 and all subsequent 
# layers unfrozen.

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.losses import BinaryCrossentropy

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "smiles/"               # Directory storing training images


# Display image_num images from input dataset
def show_images(image_num, dataset):
    class_names = dataset.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(image_num):
            ax = plt.subplot(3,3,i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    plt.show()


# Build sequential model for augmenting dataset with flipped and rotated images
def augment_data():
    
    augmenter = tf.keras.Sequential([

                    RandomFlip("horizontal"),
                    RandomRotation(0.2)

                    ])
    
    return augmenter


# Display single augmented image derived from input dataset
def show_augmented_image(dataset):
    
    # image_batch is a batch of 32 color images, and is thus of shape (32,160,160,3).
    # image is set to image_batch[0], which is the first image in the given batch. Its shape is (160,160,3).
    for image_batch, _ in dataset.take(1):
        plt.figure(figsize=(10,10))
        image = image_batch[0]                  

        # Add batch axis (of size 1) to successfully pass in single images 
        # to sequential augmenter model
        augmented_image = augmenter_model(tf.expand_dims(image, 0))

        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
        plt.show()


# Define custom model for transfer learning
# Get MobileNetv2 without dense output layer
# Add input layer for data preprocessing/augmentation
# Add, at end of MobileNetv2, pooling, dropout, dense layers
def custom_model(image_shape=IMG_SIZE, augmenter=augment_data()):
    
    input_shape = image_shape + (3,)
    
    # Get MobileNetv2 without dense output layer 
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False, 
                                                   weights="imagenet") 
    
    # Freeze MNv2 model
    base_model.trainable = False 

    # MNv2 model already has an input layer, but define another for preprocessing/augmentation
    inputs = tf.keras.Input(shape=input_shape) 
        
    # Apply data augmentation to input layer
    x = augment_data()(inputs)                 # sequential model consisting of [RandomFlip-->RandomRotation]
        
    # Preprocess with weights MNv2 model was trained on
    x = preprocessor(x) 

    # Execute call() method of base_model to use base_model in inference mode
    x = base_model(x, training=False) 
    
    # Add new layers 
    x = tfl.GlobalAveragePooling2D()(x) 
    x = tfl.Dropout(0.2)(x)
        
    # Since include_top=False for MNv2 model, there is no Dense output layer in base_model; thus add one
    outputs = tfl.Dense(1)(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model


# Plot accuracies and losses
def show_plots(history):

    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# Train custom model with MNv2 layers frozen. Custom dense output layer is unfrozen.
def train_frozen_model(model, epochs=5):

    base_lr = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

    model.save('frozen_model/')

    return history
    
    
# Unfreeze and re-train with slower learning rate
def train_unfrozen_model(model, epochs=5, lr_factor=0.1, limit=118):
    # xfer_model is an MNv2 instance applied to augmented and preprocessed inputs, with added 
    # pooling, dropout, and dense layers. Calling summary() on xfer_model shows 8 layers, but 
    # layer[4] is the MNv2 model, so xfer_model really has over 100 layers. 

    print('\nNow training unfrozen model:')

    # Freeze all "parent-level" layers in model
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze "parent-level" layer containing the 154-layer MNv2 network
    model.layers[4].trainable = True

    # Freeze, up to limit, each layer of MNv2 network
    for layer in model.layers[4].layers[:limit]:
        layer.trainable = False

    # Confirm the right layers are unfrozen
    for layer in model.layers[4].layers:
        print(layer.trainable)

    for layer in model.layers:
        print(layer.trainable)

    base_lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=(lr_factor * base_lr))

    model.compile(loss=BinaryCrossentropy(from_logits=True),
                       optimizer=optimizer,
                       metrics=["accuracy"])

    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=validation_dataset)

    model.save('unfrozen_model/')

    return history


################################
###----------Script----------###
################################

# Specification of image_size causes images to be resized to IMG_SIZE
# Despite images in smiles/ being greyscale, this function appears to automatically convert
# them into RGB images.
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)

validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)

#Call show_images() or show_augmented_image() before prefetching data
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)                                             

# Build sequential model for augmenting dataset with flipped and rotated images
augmenter_model = augment_data()

# Preprocessing
preprocessor = tf.keras.applications.mobilenet_v2.preprocess_input

# Create custom model with MNv2 network and custom input/output layers for transfer learning task
xfer_model = custom_model(IMG_SIZE, augmenter_model)

# Train only with dense output layer unfrozen
history_frozen = train_frozen_model(xfer_model)
show_plots(history_frozen)

# Train with layers starting at limit and all subsequent layers unfrozen
history_unfrozen = train_unfrozen_model(xfer_model, limit=118)
show_plots(history_unfrozen)



