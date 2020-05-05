import numpy as np

# Z-Score Normalization
def normalize_data(dataList, mu, std):

	imageList = list(dataList)

	for index in range(len(imageList)):
		imageList[index] = ( imageList[index] - mu ) / std

	return np.array(imageList)