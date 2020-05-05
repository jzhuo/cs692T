import numpy as np
from data import load_airplane_data, load_cat_data
from centralized_case import build_model, gridsearch
# keras imports
from keras.layers import Dense
from keras import regularizers, initializers, optimizers
from keras.models import Sequential, model_from_json, load_model


# run decentralized case for airplanes
def run_airplane_case_part_1(grid, numEpochs, batchSize, verbosity, hot_vector_status):
    
    x_train, y_train, x_test, y_test = load_airplane_data(hot_vector_status)

    # find best model size with gridsearch
    best_model_size = gridsearch(x_train, y_train, grid, numEpochs, batchSize, verbosity)

    # build best model and fit on data
    model = build_model(best_model_size)
    # fit the model on the training data
    model.fit(x_train, y_train, epochs=numEpochs, batch_size=batchSize, verbose=verbosity)

    # save best model to file
    if hot_vector_status:
        model.save('decentralized_airplane_model_yesVector')
    else:
        model.save('decentralized_airplane_model_noVector')

    return


# run decentralized case for cats
def run_cat_case_part_1(grid, numEpochs, batchSize, verbosity, hot_vector_status):

    x_train, y_train, x_test, y_test = load_cat_data(hot_vector_status)

    # find best model size with gridsearch
    best_model_size = gridsearch(x_train, y_train, grid, numEpochs, batchSize, verbosity)

    # build best model and fit on data
    model = build_model(best_model_size)
    # fit the model on the training data
    model.fit(x_train, y_train, epochs=numEpochs, batch_size=batchSize, verbose=verbosity)

    # save best model to file
    if hot_vector_status:
        model.save('decentralized_cat_model_yesVector')
    else:
        model.save('decentralized_cat_model_noVector')

    return


def combine_part_1(hot_vector_status):

    # load models
    if hot_vector_status:
        airplane_model = load_model('decentralized_airplane_model_yesVector')
        cat_model = load_model('decentralized_cat_model_yesVector')
    else:
        airplane_model = load_model('decentralized_airplane_model_noVector')
        cat_model = load_model('decentralized_cat_model_noVector')

    # comine models by averaging their weights
    airplane_weights = airplane_model.get_weights()
    cat_weights = cat_model.get_weights()
    
    return


if __name__ == "__main__":

    # define grid
    # grid = [512, 1024, 1536, 2048]
    grid = [1,2]

    # define hyperparameters for fitting models
    # numEpochs = 300
    numEpochs = 1
    batchSize = 10
    verbosity = 1

    # load data
    hot_vector_status = False

    # run part 1
    # run_airplane_case_part_1(grid, numEpochs, batchSize, verbosity, hot_vector_status)
    # run_cat_case_part_1(grid, numEpochs, batchSize, verbosity, hot_vector_status)
    combine_part_1(hot_vector_status)
