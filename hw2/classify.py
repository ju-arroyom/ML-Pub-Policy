import read_files
import process_data
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pylab as pl
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")

def dummify(data, col_name):
    '''
    Given a pandas dataframe and a column name
    creates dummy variables indicating whether the
    category was present or not.
    '''
    dummy = pd.get_dummies(data[col_name])
    df_add_dummy = pd.concat([data, dummy], axis =1 )
    return df_add_dummy


def do_learning(X_training, Y_training, X_test, Y_test, model_class):

    '''
    With training and testing data select the best
    features with recursive feature elimination method, then
    fit a classifier and return a tuple containing the predicted values on the test data
    and a list of the best features used.
    '''
    ref_dic = {}
    for index, x in enumerate(X_training.columns):
        ref_dic[index] = x

    model = model_class
    # Recursive Feature Elimination
    rfe = RFE(model)
    rfe = rfe.fit(X_training, Y_training)
    
    best_features = rfe.get_support(indices=True)

    best_features_names = [ref_dic[i] for i in best_features]

    predicted = rfe.predict(X_test)
    expected = Y_test

    accuracy = accuracy_score(expected, predicted)
    return (rfe, expected, predicted, best_features_names, accuracy)


def plot_confusion_matrix(data, label_list, model_name):
    '''
    Given a pandas dataframe with a confusion confusion_matrix
    and a list of axis lables plot the results
    '''
    sn.set(font_scale=1.4)#for label size

    xticks =  label_list
    yticks =  label_list
    ax = plt.axes()
    sn.heatmap(data, annot=True,annot_kws={"size": 16}, linewidths=.5, xticklabels = xticks,  
              yticklabels = yticks, fmt = '')
    ax.set_title('Confusion Matrix for' + ' ' + model_name)


def plot_roc(model_class, X_test, Y_test):
    '''
    Given the results stored in rfe and a testing
    set, the function plots the roc curve
    '''
    preds = model_class.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, preds[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()