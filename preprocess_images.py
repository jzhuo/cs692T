import numpy as np
import pickle
import os


dictionary_list = []  # each dictionary in this list is a dictionary of 10,000 values

# reading in the data
for index in range(1, 6):
	with open('data_batch_' + str(index), 'rb') as fo:
		images_dict = pickle.load(fo, encoding='bytes')
		dictionary_list.append(images_dict)

with open('test_data_batch', 'rb') as fo:
	images_dict = pickle.load(fo, encoding='bytes')
	# adding test images to the training to split later
	dictionary_list.append(images_dict)

with open('batches.meta', 'rb') as fo:
	meta_dict = pickle.load(fo, encoding='bytes')
	picture_keys = list(meta_dict.keys())
	picture_labels = meta_dict[picture_keys[picture_keys.index(b'label_names')]]

# list of the keys
keys = list(dictionary_list[0].keys()) 

# index of the keys
batch_label = keys.index(b'batch_label')
labels = keys.index(b'labels')
file_names = keys.index(b'filenames')
data = keys.index(b'data')

# ******************************************************************************************************************************
# ************************************** Filtering Different Classes of Images to File *****************************************
# ******************************************************************************************************************************

# labels for classes of data, corresponds to the label list
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
class_data_list = [list() for index in range(10)]

# parsing through the training data images
for dict_index in range(0, len(dictionary_list)):  # parsing through the list of dictionaries
	for data_index in range(0, len(dictionary_list[dict_index][keys[data]])):  # parsing through images in each dictionary
		
		data_in_row = dictionary_list[dict_index][keys[data]][data_index]  # the actual image
		picture_label_in_row = picture_labels[dictionary_list[dict_index][keys[labels]][data_index]]  # the image label
		
		# assort the training images to their specific classes
		if picture_label_in_row == picture_labels[0]:
			class_data_list[0].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[1]:
			class_data_list[1].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[2]:
			class_data_list[2].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[3]:
			class_data_list[3].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[4]:
			class_data_list[4].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[5]:
			class_data_list[5].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[6]:
			class_data_list[6].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[7]:
			class_data_list[7].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[8]:
			class_data_list[8].append(np.array(data_in_row))
		elif picture_label_in_row == picture_labels[9]:
			class_data_list[9].append(np.array(data_in_row))
		else:
			pass

# ******************************************************************************************************************************
# ***************************************** Converting Classes of Images to BW Scale *******************************************
# ******************************************************************************************************************************

# labels for classes of data, corresponds to the label list
label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

# upper indices of the colors
red = 1024       # 0 to 1023 are red
green = 2048     # 1024 to 2047 are green
blue = 3072      # 2048 to 3071 are blue

# parsing through the classes of images
for class_index in range(len(class_data_list)):

	print('Class Index:', class_index)
	
	class_of_images = class_data_list[class_index]

	print('Class of Images length:', len(class_of_images))

	# parsing through the images in the class
	for image_index in range(len(class_of_images)):

		image = class_of_images[image_index]

		#splitting up the colors
		red_list = np.array(image[:red])
		green_list = np.array(image[red:green])
		blue_list = np.array(image[green:blue])

		# declaring the new black/white image
		bw_image = []

		# converting the image RGB values to black/white greyscale value
		# each image is of size 32 x 32 = 1024
		for pixel_index in range(1024):

			# calculating the black/white value for the pixels in the image
			weighted_average = (.229 * red_list[pixel_index]) + (.587 * green_list[pixel_index]) + (.114 * blue_list[pixel_index])
			bw_image.append(weighted_average)
		
		# type casting the image to a numpy array
		bw_image = np.array(bw_image)
		
		# saving the black/white images into class of images
		class_of_images[image_index] = bw_image

	print('length of first image', len(class_of_images[0]))
	print('values of first image', class_of_images[0])
	print()

	# # saving the black/white class of images into data list
	# class_data_list[class_index] = class_of_images

	# saving class of images to file based on label list
	f = open('bw_' + label_list[class_index] + '_images', 'wb')
	pickle.dump(class_of_images, f)
	f.close()
