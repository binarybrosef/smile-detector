# smile-detector
A simple convolutional neural network that classifies face-containing images as smiling or not smiling.

## The Challenge
A variety of automated systems have been developed that take action contingent upon a human subject's facial expression as determined by imaging their face. In some examples, entry to a facility, location, or experience - such as an amusement park or ride therein - may depend upon human subjects exhibiting particular facial expression(s) in view of a camera. 

This convolutional neural network (CNN) is configured to detect whether human faces in images are smiling or not smiling. This task is inspired by an assignment that is part of the convolutional neural networks course of Coursera/DeepLearning.AI's [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning). 

## The Data
This CNN is trained on a training set provided by Coursera/DeepLearning.AI consisting of 600 64x64 RGB images. Each image contains a human face and appears to have been captured with the same camera in the same location. A test set consisting of 150 examples is further provided with which to evaluate model performance.

Unfortunately, the dataset provided by Coursera/DeepLearning.AI appears to be proprietary and only accessible by obtaining access to the Deep Learning Specialization. As such, this dataset is not provided as part of this repository.  However, models can be evaluated against any face-containing RGB image that is of size 64x64. 

Given the narrow distribution formed by this dataset - images taken with the same camera in the same location, possibly under a relatively common set of lighting conditions and distances between subject and camera - the CNN, when merely trained on this dataset, tends to generalize poorly. As such, the CNN is further trained on a [supplemental dataset](https://github.com/hromi/SMILEsmileD) consisting of 64x64 greyscale images (9476 negative examples, 3690 positive examples).

## The Model
A CNN of nearly minimial simplicity is employed to classify human faces in images as smiling or not similing. The CNN consists of the following layers:
- zero-padding (3 pixels) 
- 2D convolution (32 filters, kernel size of 7, stride of 1)
- batch normalization (along color channels axis)
- ReLU activation
- 2D max pooling
- flattening
- dense with sigmoid activation

The CNN is constructed using TensorFlow's Sequential API. Data processing merely consists of scaling pixel values. 

`model.py` constructs, trains, saves, and evaluates the CNN. A saved and trained version of the model is provided in the `model` folder.

## Inferencing on Unseen Examples
`inference.py` enables the CNN to be evaluated on user-provided images. To evaluate the CNN on your own images, place images of any size in the `images` folder. Delete the README file before inferencing on these images. `inference.py` resizes, via `resize()`, these images to 64x64 and places the resized images in a new folder named `resized_images`. `inference()` is then called, which loads the trained version of the CNN from the `model` folder, applies the CNN to the images in `resized_images`, and outputs classifications for each image as containing a face that is either smiling or not smiling. 

## Limitations and Future Development
While the CNN performs well (approximately 95% accuracy) on the test set provided by Coursera/DeepLearning.AI, suboptimal performance was observed when applying the CNN to a random set of face-containing images obtained via internet image search. As such, the CNN is trained on supplemental data to achieve improved generalization. 

Further gains in performance could likely be obtained by expanding the training set with randomly chosen images that form a wider distribution to which the CNN can better generalize. It is possible desired performance could be obtained according to this strategy while keeping CNN topology relatively the same. This could result in a model that achieves desired performance with very low computational cost, and thus could be suitable to contexts in which compute resources are constrained (e.g., mobile, low-power/battery-powered contexts). However, as the current CNN topology is very simple, even slight increases in topology complexity could notably increase performance. 
