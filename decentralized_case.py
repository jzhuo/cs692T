import numpy as np
from data import load_airplane_data, load_cat_data
from centralized_case import build_model, gridsearch
# keras imports
from keras.layers import Dense
from keras import regularizers, initializers, optimizers
from keras.models import Sequential, model_from_json, load_model


# run decentralized case for airplanes
def run_airplane_case_1(grid, numEpochs, batchSize, verbosity, hot_vector_status):
    
    x_train, y_train, x_test, y_test = load_airplane_data(hot_vector_status)

    # find best model size with gridsearch
    best_model_size = gridsearch(x_train, y_train, grid, numEpochs, batchSize, verbosity)

    # build best model and fit on data
    model = build_model(best_model_size)

    # fit the model on the training data
    model.fit(x_train, y_train, epochs=numEpochs, batch_size=batchSize, verbose=verbosity)

    # save best model to file
    if hot_vector_status:
        model.save('decentralized_airplane_model_1_yesVector')
    else:
        model.save('decentralized_airplane_model_1_noVector')

    return


# run decentralized case for cats
def run_cat_case_1(grid, numEpochs, batchSize, verbosity, hot_vector_status):

    x_train, y_train, x_test, y_test = load_cat_data(hot_vector_status)

    # find best model size with gridsearch
    best_model_size = gridsearch(x_train, y_train, grid, numEpochs, batchSize, verbosity)

    # build best model and fit on data
    model = build_model(best_model_size)

    # fit the model on the training data
    model.fit(x_train, y_train, epochs=numEpochs, batch_size=batchSize, verbose=verbosity)

    # save best model to file
    if hot_vector_status:
        model.save('decentralized_cat_model_1_yesVector')
    else:
        model.save('decentralized_cat_model_1_noVector')

    return


def combine_weights(weightsA, weightsB):

    # update weightsA to be the new averaged weights between A and B
    for index in range(len(weightsA)):

        # each weights matrix has 4 components, index goes from 0 to 3
        a = weightsA[index]
        b = weightsB[index]

        # each component can either have a shape of 2 dimensions or just 1 dimension
        if len(a.shape) == 1:
            for i in range(len(a)):
                # update flat list elements with average
                a[i] = (a[i] + b[i])/2
            weightsA[index] = a
        elif len(a.shape) == 2:
            # update every row/column element with average
            for rowIndex in range(len(a)):
                for colIndex in range(len(a[rowIndex])):
                    a[rowIndex][colIndex] = (a[rowIndex][colIndex] + b[rowIndex][colIndex])/2
            weightsA[index] = a
        else:
            print('*************************************************************')
            print('*************************************************************')
            print('*************************************************************')
            print('******************** ERROR: CODE FAILURE ********************')
            print('*************************************************************')
            print('*************************************************************')
            print('*************************************************************')
            return None

    return weightsA


def combine_models(stepNum, hot_vector_status):

    # load models
    if hot_vector_status:
        airplane_model = load_model('decentralized_airplane_model_'+str(stepNum)+'_yesVector')
        cat_model = load_model('decentralized_cat_model_'+str(stepNum)+'_yesVector')
    else:
        airplane_model = load_model('decentralized_airplane_model_'+str(stepNum)+'_noVector')
        cat_model = load_model('decentralized_cat_model_'+str(stepNum)+'_noVector')

    # comine models by averaging their weights
    airplane_weights = np.array(airplane_model.get_weights())
    cat_weights = cat_model.get_weights()
    combined_weights = combine_weights(airplane_weights, cat_weights)

    # update model: load new weights into it
    airplane_model.set_weights(combined_weights)
    # save best model to file
    if hot_vector_status:
        airplane_model.save('combined_model_'+str(stepNum)+'_yesVector')
    else:
        airplane_model.save('combined_model_'+str(stepNum)+'_noVector')

    return


# run decentralized case part 2 for airplanes
def run_airplane_case_2(stepNum, numEpochs, batchSize, verbosity, hot_vector_status):
    
    # load data
    x_train, y_train, x_test, y_test = load_airplane_data(hot_vector_status)

    # build best model and fit on data
    if hot_vector_status:
        model = load_model('combined_model_'+str(stepNum-1)+'_yesVector')
    else:
        model = load_model('combined_model_'+str(stepNum-1)+'_noVector')

    # fit the model on the training data
    model.fit(x_train, y_train, epochs=numEpochs, batch_size=batchSize, verbose=verbosity)

    # save best model to file
    if hot_vector_status:
        model.save('decentralized_airplane_model_'+str(stepNum)+'_yesVector')
    else:
        model.save('decentralized_airplane_model_'+str(stepNum)+'_noVector')

    return


# run decentralized case part 2 for cats
def run_cat_case_2(stepNum, numEpochs, batchSize, verbosity, hot_vector_status):
    
    # load data
    x_train, y_train, x_test, y_test = load_cat_data(hot_vector_status)

    # build best model and fit on data
    if hot_vector_status:
        model = load_model('combined_model_'+str(stepNum-1)+'_yesVector')
    else:
        model = load_model('combined_model_'+str(stepNum-1)+'_noVector')

    # fit the model on the training data
    model.fit(x_train, y_train, epochs=numEpochs, batch_size=batchSize, verbose=verbosity)

    # save best model to file
    if hot_vector_status:
        model.save('decentralized_cat_model_'+str(stepNum)+'_yesVector')
    else:
        model.save('decentralized_cat_model_'+str(stepNum)+'_noVector')

    return


if __name__ == "__main__":

    # define grid
    grid = [512, 1024, 1536, 2048]

    # define hyperparameters for fitting models
    numEpochs = 300
    batchSize = 10
    verbosity = 1

    # load data
    hot_vector_status = False

    # run part 1
    run_airplane_case_1(grid, numEpochs, batchSize, verbosity, hot_vector_status)
    run_cat_case_1(grid, numEpochs, batchSize, verbosity, hot_vector_status)

    # combine weights of both models to get new model
    combine_models(1, hot_vector_status)

    # run part 2
    run_airplane_case_2(2, numEpochs, batchSize, verbosity, hot_vector_status)
    run_cat_case_2(2, numEpochs, batchSize, verbosity, hot_vector_status)

    # combine models for a second time
    combine_models(2, hot_vector_status)

    # run part 3
    run_airplane_case_2(3, numEpochs, batchSize, verbosity, hot_vector_status)
    run_cat_case_2(3, numEpochs, batchSize, verbosity, hot_vector_status)

    # combine models for a third time
    combine_models(3, hot_vector_status)
