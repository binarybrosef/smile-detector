##--Script for applying model to user-supplied images. Place images in \\images directory to 
##--be resized. Resized images are placed in \\resized_images. Model is then evaluated on every
##--image in \\resized_images.

from PIL import Image
import os
import numpy as np
from tensorflow.keras.models import load_model

IMAGE_PATH = os.getcwd() + '\\images'				#Path to directory storing user-supplied images
RESIZE_PATH = os.getcwd() + '\\resized_images'		#Path to directory storing resized images

#Resize images in \\images directory to 64x64 RGB images; save in \\resized_images directory
def resize():

	image_list = os.listdir(IMAGE_PATH)

	try:
		os.mkdir(os.getcwd() + '\\resized_images')
	except FileExistsError:
		print('resized_images directory already exists')

	for image in image_list:
		img = Image.open(IMAGE_PATH + '\\' + image)
		img_resized = img.resize((64, 64))

		img_resized.save(RESIZE_PATH + '\\' + image)

#Apply model to resized images in \\resized_images
def inference(model):

	image_list = os.listdir(RESIZE_PATH)
	images_array = np.empty((1,64,64,3))

	i = 0
	for image in image_list:
		img = Image.open(RESIZE_PATH + '\\' + image)
		img_array = np.array(img)
		img_array = img_array/255.

		images_array[0] = img_array
		prediction = model(images_array)

		if prediction >= 0.5:
			print(f'{image} is classified as smiling at {prediction}')
		else:
			print(f'{image} is classified as not smiling at {prediction}')

		i += 1

	
#Load trained model from \\model directory and inference on images in \\resized_images
r_model = load_model('model')
inference(r_model)
