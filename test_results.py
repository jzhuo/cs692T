import numpy as np
from centralized_case import build_model, gridsearch
from data import load_centralized_data, load_airplane_data, load_cat_data
# keras imports
from keras.layers import Dense
from keras import regularizers, initializers, optimizers
from keras.models import Sequential, model_from_json, load_model


def check_pred(predLabel, trueLabel):

    if predLabel[0] >= .5:
        predLabel[0] = 1
    else:
        predLabel[0] = 0

    if predLabel[1] >= .5:
        predLabel[1] = 1
    else:
        predLabel[1] = 0

    if predLabel[0] == trueLabel[0] and predLabel[1] == trueLabel[1]:
        return 1
    else:
        return 0


if __name__ == "__main__":

    # ************************************************************************
    # decentralized model accuracy
    # ************************************************************************

    a_x_train, a_y_train, a_x_test, a_y_test = load_airplane_data(False)
    c_x_train, x_y_train, c_x_test, x_y_test = load_cat_data(False)

    combinedModel1 = load_model('combined_model_1_noVector')
    combinedModel2 = load_model('combined_model_2_noVector')
    combinedModel3 = load_model('combined_model_3_noVector')

    a1_pred = combinedModel1.predict(a_x_test)
    c1_pred = combinedModel1.predit(c_x_test)

    a2_pred = combinedModel2.predict(a_x_test)
    c2_pred = combinedModel2.predit(c_x_test)

    a3_pred = combinedModel3.predict(a_x_test)
    c3_pred = combinedModel3.predit(c_x_test)

    a1_sum, c1_sum, a2_sum, c2_sum, a3_sum, c3_sum = 0, 0, 0, 0, 0, 
    
    for index in range(len(a1_pred)):

        c1_sum += check_pred(c1_pred[index],c_y_test[index])
        a2_sum += check_pred(a2_pred[index],a_y_test[index])
        c2_sum += check_pred(c2_pred[index],c_y_test[index])
        a3_sum += check_pred(a3_pred[index],a_y_test[index])
        c3_sum += check_pred(c3_pred[index],c_y_test[index])

    a1_acc = a1_sum/len(a1_pred)
    c1_acc = c1_sum/len(c1_pred)
    a2_acc = a2_sum/len(a2_pred)
    c2_acc = c2_sum/len(c2_pred)
    a3_acc = a3_sum/len(a3_pred)
    c3_acc = c3_sum/len(c3_pred)

    combined1_acc = (a1_acc + c1_acc)/2
    combined2_acc = (a2_acc + c2_acc)/2
    combined3_acc = (a3_acc + c3_acc)/2

    print('\n\n ******************************************')
    print('Decentralized Combined Model 1 Results\n')
    print('airplane accuracy:', a1_acc)
    print('cat accuracy:', c1_acc)
    print('combined accuracy:', combined1_acc)
    print('******************************************\n')

    print('\n\n ******************************************')
    print('Decentralized Combined Model 2 Results\n')
    print('airplane accuracy:', a2_acc)
    print('cat accuracy:', c2_acc)
    print('combined accuracy:', combined2_acc)
    print('******************************************\n')

    print('\n\n ******************************************')
    print('Decentralized Combined Model 3 Results\n')
    print('airplane accuracy:', a3_acc)
    print('cat accuracy:', c3_acc)
    print('combined accuracy:', combined3_acc)
    print('******************************************\n')


    # ************************************************************************
    # centralized model accuracy
    # ************************************************************************

    centralizedModel = load_model('centralized_model_noVector')

    centralizedPredAirplane = centralizedModel.predict(a_x_test)
    centralizedPredCat = centralizedModel.predict(c_x_test)

    centralizedSumAirplane, centralizedSumCat = 0, 0
    for index in range(len(centralizedPred)):

        centralizedSumAirplane += check_pred(centralizedPredAirplane[index],a_y_test[index])
        centralizedSumCat += check_pred(centralizedPredCat[index],c_y_test[index])

    centralized_airplane_acc = centralizedSumAirplane/len(centralizedPredAirplane)
    centralized_cat_acc = centralizedSumCat/len(centralizedPredCat)

    centralized_combined_acc = (centralized_airplane_acc + centralized_cat_acc)/2


    print('\n\n ******************************************')
    print('Centralized Model Results\n')
    print('airplane accuracy:', centralized_airplane_acc)
    print('cat accuracy:', centralized_cat_acc)
    print('combined accuracy:', centralized_combined_acc)
    print('******************************************\n')