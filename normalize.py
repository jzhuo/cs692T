import numpy as np

# Z-Score Normalization
def normalize_data(dataList):

	imageList = np.array(dataList)
	
	mu = np.mean(imageList)
	std = np.std(imageList)

	imageList = list(imageList)

	for index in range(len(imageList)):
		imageList[index] = ( imageList[index] - mu ) / std

	return np.array(imageList)