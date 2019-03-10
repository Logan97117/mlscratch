#Adaboost classifier from scratch implemented on lending club dataset
#libaries
import numpy as np
import math
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def adaboost_fit(training_data,features_train,targets_train,num_clfs,max_depth_tree):
    x_train = features_train.values
    y_train = targets_train.values
    weights = np.empty(shape = (num_clfs,),dtype=float) # belief weights for classifiers
    alpha = np.array(list([float(1/features_train.shape[0]) for i in range(features_train.shape[0])])) # weights of data points
    clfs = list([DecisionTreeClassifier(max_depth = max_depth_tree) for c in range(num_clfs)])
    
    #training weights of the weak classifiers, each classifier is trained for iters number of iterations
    for i in range(num_clfs):
        clfs[i].fit(x_train,y_train)
        predictions = clfs[i].predict(x_train)
        weighted_error = compute_weighted_error(alpha,y_train,predictions)
        weights[i] = 0.5*math.log((1-weighted_error)/weighted_error)
        normalized_computed_data_point_weights(weights[i],alpha,y_train,predictions)
        
        
    return weights,clfs    


    
def predict(clf_weights,classifiers,data):
    y_pred_score = np.zeros(shape = (data.shape[0],),dtype = float)
    for i in range(len(classifiers)):
        y_pred_score = y_pred_score + (clf_weights[i] * classifiers[i].predict(data.values))
        
    predictions = np.sign(y_pred_score)
    return predictions
    

def compute_weighted_error(weights_data_points,y_train,y_pred):
    sum = 0
    for i in range(len(y_train)):
        if(y_train[i]!=y_pred[i]):
            sum = sum + weights_data_points[i]


    return (sum/np.sum(weights_data_points))        

def normalized_computed_data_point_weights(clf_weight,data_point_weights,y_train,y_pred):
    for i in range(len(data_point_weights)):
        if(y_train[i]==y_pred[i]):
            data_point_weights[i] = data_point_weights[i] * math.exp(-clf_weight)

        else:
            data_point_weights[i] = data_point_weights[i] * math.exp(clf_weight)

    data_point_weights = data_point_weights/float(np.sum(data_point_weights))   

       


