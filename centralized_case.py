import numpy as np
from data import load_centralized_data
# keras imports
from keras.layers import Dense
from keras import regularizers, initializers, optimizers
from keras.models import Sequential, model_from_json, load_model


# build a model with pre-defined hyperparameters with hidden layer size specified
def build_model(hidden_layer_size):

    # initializer, with fan-in
    minValue = 1 / hidden_layer_size - 1e-4
    maxValue = 1 / hidden_layer_size + 1e-4
    # seed = 7 for consistency and testing purposes
    fan_in_init = initializers.RandomUniform(
        minval=minValue, maxval=maxValue, seed=7)

    # regularizer, using L2 norm
    l2_regularization = regularizers.l2(1e-6)

    # optimizer
    adam = optimizers.Adam(
        lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0, amsgrad=False
    )

    model = Sequential()
    # first hidden layer
    model.add(
        Dense(
            hidden_layer_size, 
            activation='relu', 
            use_bias=True, 
            kernel_initializer=fan_in_init, 
            bias_initializer='zeros', 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None
        )
    )
    # output layer
    model.add(
        Dense(
            2, 
            activation='sigmoid', 
            use_bias=True, 
            kernel_initializer=fan_in_init, 
            bias_initializer='zeros', 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None
        )
    )

    # compile model with optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    return model


# search through multiple models to find optimal hidden layer size
def gridsearch(x_train, y_train, grid, numEpochs, batchSize, verbosity):

    # split the training data into smaller subset of training and testing
    cutoff = int(len(x_train[0])*.9)
    x_train_train = x_train[0][:cutoff]
    y_train_train = y_train[0][:cutoff]
    x_train_test = x_train[0][cutoff:]
    y_train_test = y_train[0][cutoff:]

    models = []
    accuracyList = []
    for hiddenSize in grid:
        # build model
        model = build_model(hiddenSize)
        # save model
        models.append(model)
        # fit model to data
        model.fit(x_train_train, y_train_train, epochs=numEpochs, batch_size=batchSize, verbose=verbosity)

        # predict with model and clean up prediction for classification results analysis
        predictions = model.predict(x_train_test)
        predictionList = []
        for prediction in predictions:
            _ = []
            for value in prediction:
                if value >= .5:
                    _.append(1)
                else:
                    _.append(0)
            predictionList.append(_)
        
        # compute accuracy of model and save to accuracyList
        correctPredictions = 0
        for index in range(len(predictionList)):
            prediction = predictionList[index]
            classLabel = list(y_train_test[index])

            if prediction[0] == classLabel[0] and prediction[1] == classLabel[1]:
                correctPredictions += 1
        accuracyList.append(correctPredictions/len(predictionList))    

    # find model with highest accuracy
    best_model_size = grid[accuracyList.index(max(accuracyList))]
    
    return best_model_size


if __name__ == "__main__":

    # define grid
    grid = [512, 1024, 1536, 2048]

    # define hyperparameters for fitting models
    numEpochs = 300
    batchSize = 10
    verbosity = 1

    # load data
    hot_vector_status = False
    x_train, y_train, x_test, y_test = load_centralized_data(hot_vector_status)

    # find best model size with gridsearch
    best_model_size = gridsearch(x_train, y_train, grid, numEpochs, batchSize, verbosity)

    # build best model and fit on data
    model = build_model(best_model_size)
    # fit the model on the training data
    model.fit(x_train, y_train, epochs=numEpochs, batch_size=batchSize, verbose=verbosity)

    # save best model to file
    model.save('centralized_model')