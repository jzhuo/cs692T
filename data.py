import pickle
import random
from normalize import normalize_data
import numpy as np


def load_centralized_data(hot_vector):

    labels = ['airplane', 'cat']
    labelClasses = [[1,0], [0,1]] 

    # save image data to dataList
    dataList = []
    for x in range(len(labels)):
        label = labels[x]
        labelClass = labelClasses[x]
        with open('bw_' + label + '_images', 'rb') as f:
            data = pickle.load(f)
            data = normalize_data(data)
        if hot_vector:
            for y in range(len(data)):
                # add hot vectors to the images
                data[y] = np.array(labelClass + list(data[y]))
        dataList.append(data)

    # generate and save corresponding label data to labelsList
    labelsList = []
    # first class
    labelsList.append(
        [[1, 0] for x in range(len(dataList[0]))]
    )
    # second class
    labelsList.append(
        [[0, 1] for x in range(len(dataList[0]))]
    )

    # split into training and testing data
    x_train = list(dataList[0][:5400]) + list(dataList[1][:5400])
    y_train = list(labelsList[0][:5400]) + list(labelsList[1][:5400])
    x_test = list(dataList[0][5400:]) + list(dataList[1][5400:])
    y_test = list(labelsList[0][5400:]) + list(labelsList[1][5400:])

    # shuffle x_train and y_train together
    a = list(zip(x_train, y_train)) 
    random.shuffle(a) 
    x_train, y_train = zip(*a) 
    # shuffle x_test and y_test together
    b = list(zip(x_test, y_test)) 
    random.shuffle(b) 
    x_test, y_test = zip(*b) 

    return np.array([x_train]), np.array([y_train]), np.array([x_test]), np.array([y_test])


def load_airplane_data(hot_vector):

    with open('bw_airplane_images', 'rb') as f:
        data = pickle.load(f)
        dataList = normalize_data(data)

    # generate and save corresponding label data to labelsList
    labelsList = [[1, 0] for x in range(len(dataList))]

    # split into training and testing data
    x_train = list(dataList[:5400])
    y_train = list(labelsList[:5400])
    x_test = list(dataList[5400:])
    y_test = list(labelsList[5400:])

    # shuffle x_train and y_train together
    a = list(zip(x_train, y_train)) 
    random.shuffle(a) 
    x_train, y_train = zip(*a) 
    # shuffle x_test and y_test together
    b = list(zip(x_test, y_test)) 
    random.shuffle(b) 
    x_test, y_test = zip(*b) 

    return np.array([x_train]), np.array([y_train]), np.array([x_test]), np.array([y_test])


def load_cat_data(hot_vector):

    with open('bw_cat_images', 'rb') as f:
        data = pickle.load(f)
        dataList = normalize_data(data)

    # generate and save corresponding label data to labelsList
    labelsList = [[0, 1] for x in range(len(dataList))]

    # split into training and testing data
    x_train = list(dataList[:5400])
    y_train = list(labelsList[:5400])
    x_test = list(dataList[5400:])
    y_test = list(labelsList[5400:])

    # shuffle x_train and y_train together
    a = list(zip(x_train, y_train)) 
    random.shuffle(a) 
    x_train, y_train = zip(*a) 
    # shuffle x_test and y_test together
    b = list(zip(x_test, y_test)) 
    random.shuffle(b) 
    x_test, y_test = zip(*b) 

    return np.array([x_train]), np.array([y_train]), np.array([x_test]), np.array([y_test])


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_centralized_data(False)

    print('centralized data')
    print(x_train.shape, x_train)
    print(y_train.shape, y_train)
    print(x_test.shape, x_test)
    print(y_test.shape, y_test)
    print('\n\n\n')

    x_train, y_train, x_test, y_test = load_airplane_data(False)

    print('airplane data')
    print(x_train.shape, x_train)
    print(y_train.shape, y_train)
    print(x_test.shape, x_test)
    print(y_test.shape, y_test)
    print('\n\n\n')

    x_train, y_train, x_test, y_test = load_cat_data(False)

    print('cat data')
    print(x_train.shape, x_train)
    print(y_train.shape, y_train)
    print(x_test.shape, x_test)
    print(y_test.shape, y_test)
    print('\n\n\n')