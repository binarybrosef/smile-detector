# smile-detector
Applying transfer learning to a MobileNetv2 for binary image classification.

## Objective
A variety of systems have been developed for classifying human facial expression in images. For example, automated systems exist that take action contingent upon a human subject's facial expression, such as selectively granting access to a facility, location, or experience (e.g., amusement park or ride therein). The classification of facial expression may be sought for many other reasons, such as mood classification. 

This repository provides convolutional neural networks (CNNs) configured to detect whether human faces in images are smiling or not smiling. CNNs are built from a 154-layer MobileNetv2 pre-trained on ImageNet. 

## MobileNetv2
MobileNetv2 is a CNN architecture that has gained popularity for its ability to achieve good performance on image data with low computational cost. Identifying features of the MobileNetv2 architecture include bottleneck layers, residual connections, and depthwise convolutions. 

## Transfer Learning and Training Data
Transfer learning is performed using a training dataset consisting of 64x64 greyscale face-containing images (9476 negative examples, 3690 positive examples). The dataset is publicly accessible from this [Github repository](https://github.com/hromi/SMILEsmileD).

Two different approaches to transfer learning are explored to ascertain the levels of performance that these approaches can achieve:
- In the first approach, transfer learning on the training dataset is performed with only the final, dense output layer of a pre-trained MobileNetv2 unfrozen. This is achieved by importing a MobileNetv2 model with `include_top=False` and adding a new, trainable dense output layer to the model.
- In the second approach, transfer learning on the training dataset is performed with an adjustable number of final layers in a pre-trained MobileNetv2 unfrozen. By default, layer 118 and all subsequent layers are unfrozen. This is achieved by setting the `trainable` attribute of the unfrozen layers to `True`. A reduced learning rate is used in the second approach as compared to the first approach.

## Data Augmentation
To improve model performance, data augmentation is performed by randomly flipping and rotating training images. The `augment_data()` function returns a sequential Tensorflow model comprising `RandomFlip` and `RandomRotation` layers for this purpose.

## Model Construction and Script Use
`script.py` implements transfer learning of a MobileNetv2 according to the two approaches described above. 
`custom_model()` returns a functional Tensorflow model comprising a MobileNetv2 pre-trained on ImageNet without its dense output layer, a data augmentation layer by way of `augment_data()`, a preprocessing layer implemented by `tf.keras.applications.mobilenet_v2.preprocess_input`, and "custom" layers at the end of the model consisting of `GlobalAveragePooling2D`, `Dropout`, and `Dense` layers. 

**Important note:** the output of `custom_model()` is a "nested" model that appears to have only eight layers. However, the fifth layer (layer[4]) contains a      154-layer MobileNetv2 instance. Thus, the individual layers of the MobileNetv2 instance can be accessed via `custom_model.layer[4].layers` if `custom_model` is the output from `custom_model()`.

`train_frozen_model()` trains a model produced by `custom_model()` - i.e., a pre-trained MobileNetv2 with the aforementioned custom input/output layers - with only the custom dense output layer unfrozen and all other layers frozen.

`train_unfrozen_model()` trains a model produced by `custom_model()` with the layer set by the `limit` keyword argument and all subsequent layers unfrozen. By default, limit = layer 118 of the MobileNetv2.

A trained version of the model produced by `train_frozen_model()` is saved in the `frozen_model` directory, and a trained version of the model produced by `train_unfrozen_model()` is saved in the `unfrozen_model` directory.

`script.py` also provides the functions `show_images()` for displaying a selected number of training images, `show_augmented_image()` for displaying a single augmented training image, and `show_plots()` for visualizing accuracies and losses for the two transfer learning approaches.

## Results
Losses and accuracies for the training and validation (chosen as 20% of the overall input dataset) datasets are summarized below. For the "frozen" approach in which only the dense output layer is unfrozen:
![frozen_model](https://user-images.githubusercontent.com/491395/164129693-fe1eee13-9e73-499f-87e2-bef8729b237e.png)

For the "unfrozen" approach in which layer 118 and all layers onward in the MobileNetv2 instance are unfrozen:
![unfrozen_model](https://user-images.githubusercontent.com/491395/164129706-51b71bb2-e9e8-4e90-b4d3-83fe625f8657.png)

As can be seen from the plots above, the unfrozen approach achieves noticeably superior performance (as measured by validation accuracy) over the frozen approach by the first epoch. Best performance by the unfrozen approach, among five epochs, is achieved by epoch 3, with a validation accuracy of approximately 93%. 

By comparison, the frozen approach achieves a validation accuracy of approximately 86% at epoch 3, although increasing performance is observed with increasing epoch number. It is possible the frozen approach could achieve comparable performance with the unfrozen approach if training is continued past five epochs.

I speculate that the unfrozen approach achieves nearly optimal performance by epoch 1 due to the relative simplicity and low size of the training dataset. Unfreezing the model under this approach would appear to provide a large parameter space with which a good fit to the training set can be achieved in a low amount of training time. 

## Limitations and Future Development
As mentioned above, improved performance could likely be achieved with the frozen approach by simply training for more epochs. For the unfrozen approach, greater performance is likely obtainable by varying which layers of the MobileNetv2 are frozen/unfrozen. A greater number of training epochs, and variation of which layers are frozen/unfrozen, was not performed due to limitations of the hardware on which this script was executed. 

While the generalization ability of models was not a focus of this project, greater generalization performance beyond what the models can currently achieve could likely be obtained by expanding the training set with randomly chosen images that form a wider distribution to which the models can better generalize. For example, the training set could be expanded with color and/or higher-resolution images.
