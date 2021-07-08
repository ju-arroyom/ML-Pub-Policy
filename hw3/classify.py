# Import pre-processing and plotting libraries
import time
import read_files
import process_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pylab as pl
# Import models and metrics from sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import ParameterGrid, train_test_split

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.classification import _check_targets



import warnings
warnings.filterwarnings("ignore")


def define_clfs_params(grid_size):

    """
    Defines classifiers and param grid that are used
    to find best model.
    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    """

    clfs = {
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'DT': DecisionTreeClassifier(),
        'LSVC': svm.LinearSVC(penalty='l1', random_state=0, dual=False), 
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'NB': GaussianNB(),  
            }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [5]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LSVC' :{'penalty': ['l1','l2'], 'C' :[0.001,0.01,0.1,1,10]},
    'KNN' :{'n_neighbors': [1,5,10,25],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LSVC' :{'C' :[0.01]},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

        



def clf_loop(models_to_run, clfs, grid, X, y , print_plots = False):
    """
    Loops through classifiers and stores metrics in pandas df.
    Df gets returned.

    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    In:
        - models_to_run: (list) of models to run
        - clfs: (dict) of classifiers
        - grid: (dict) of classifiers with set of parameters to train on
        - X_train: features from training set
        - X_test: features from test set
        - y_train: targets of training set
        - y_test: targets of test set
        - print_plots: (bool) whether or not to print plots
    Out:
        - pandas df
    """

    conf_matrix = {}
    
    count = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    results_df = pd.DataFrame(columns=('model_type', 'clf', 'parameters', 'train_time',
                                        'predict_time', 'auc-roc', 'p_at_5', 'p_at_10',
                                        'p_at_20', 'r_at_5', 'r_at_10', 'r_at_20',
                                        'f1_at_5', 'f1_at_10', 'f1_at_20'))
    
    X_train = rf_imputation(X_train)
    X_train = feature_eng(X_train)
    
    X_test = rf_imputation(X_test)
    X_test = feature_eng(X_test)
    

    for index, clf in enumerate([clfs[x] for x in models_to_run]):

        print(models_to_run[index])

        parameter_values = grid[models_to_run[index]]

        for p in ParameterGrid(parameter_values):

            try:
                clf.set_params(**p)

                start_time_training = time.time()
                model = clf.fit(X_train, y_train)
                train_time = time.time() - start_time_training

                start_time_predicting = time.time()

                if models_to_run[index] == 'LSVC':

                    y_pred_probs = model.decision_function(X_test)

                else:
                    y_pred_probs = model.predict_proba(X_test)[:,1]


                predict_time = time.time() - start_time_predicting

                roc_score = roc_auc_score(y_test, y_pred_probs)

                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))

                pr_5, r_5, f1_5 = metrics_at_k(y_test_sorted,y_pred_probs_sorted,5.0)
                pr_10, r_10, f1_10 = metrics_at_k(y_test_sorted,y_pred_probs_sorted,10.0)
                pr_20, r_20, f1_20 = metrics_at_k(y_test_sorted,y_pred_probs_sorted, 20.0)

                results_df.loc[len(results_df)] = [models_to_run[index], clf, p,
                                                   train_time, predict_time, roc_score, pr_5, pr_10,
                                                   pr_20, r_5, r_10, r_20, f1_5, f1_10, f1_20]

                
                cm = confusion_matrix(sorted(y_test, reverse=True), sorted(generate_binary_at_k(y_test, 20.0), reverse=True))

                if count not in conf_matrix:
                    conf_matrix[count] = {'matrix' : cm , 'title': models_to_run[index]}
                 
                count = count + 1
                
                if print_plots:

                    plot_precision_recall_n(y_test, y_pred_probs, clf)

            except IndexError as e:
                print('Error:', e)
                continue

    return results_df, conf_matrix


def generate_binary_at_k(y_scores, k):
    '''
    Generate binary at threshold k
    Inputs: y_scores predict proba from test target
            k : threshold
    Returns: test predicitions
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary



def metrics_at_k(y_true, y_scores, k):
    '''
    Calculate metrics at threshold k.
    The metrics are precision, recall, and f1.
    Inputs: y_true : target from test dataframe
            y_scores: predict proba from test target
            k: user defined threshold

    Returns: Tuple with precision, recall and f1 
             for threshold k
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)

    return precision, recall, f1




def plot_precision_recall_n(y_true, y_prob, model_name):

    """
    Function to plot precision recall curve.
    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    """

    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)

    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)

    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()




def plot_confusion_matrix(data, label_list, model_name):
    '''
    Given a pandas dataframe with a confusion confusion_matrix
    and a list of axis lables plot the results
    '''
    sns.set(font_scale=1.4)#for label size

    xticks =  label_list
    yticks =  label_list
    ax = plt.axes()
    sns.heatmap(data, annot=True,annot_kws={"size": 16}, linewidths=.5, xticklabels = xticks,  
              yticklabels = yticks, fmt = '')
    ax.set_title('Confusion Matrix for' + ' ' + model_name)


def confusion_party(matrix_dictionary, label_list):
    '''
    Produces the subplots with the confusion matrices
    for each model and set of parameters.
    Inputs: A dictionary containing the confusion
            matrices for the models
            A list of labels for the axis
    Returns:
            Confusion party 
    '''
    import math
    fix, ax = plt.subplots(figsize=(16, 12))
    plt.suptitle('Confusion Matrix of Various Classifiers')
    for key, values in matrix_dictionary.items():
        matrix = values['matrix']
        title = values['title']
        plt.subplot(3, math.ceil(len(matrix_dictionary)/3), key) # starts from 1
        plt.title(str(key)+ " : " +title);
        xticks =  label_list
        yticks =  label_list
        sns.heatmap(matrix, annot=True,annot_kws={"size": 16}, linewidths=.5, xticklabels = xticks,  
                  yticklabels = yticks, fmt = '');


def do_learning(models_to_run, grid_size, X_data, target):
    '''
    Run the list of classifiers given the features provided by the user.
    '''
    clfs, grid = define_clfs_params(grid_size)

    results_df, cm_dic = clf_loop(models_to_run, clfs,grid, X_data, target)

    results_df.to_csv('results.csv', index=False)

    return results_df, cm_dic


def rf_imputation(dataframe):
    """
    Impute missing income values with 
    RandomForest Classifier
    """
    new_df = dataframe.drop(['dummy', 'personid'], axis=1)
    process_data.impute_val_to_column(new_df,'numberofdependents', 'random', 0, 3, [0.65,0.2,0.15])
    # One age value is missing
    new_df['age'].replace(0, new_df['age'].mean(), inplace = True)
    # Condition for Nan
    cond = new_df["monthlyincome"].isna()
    xtrain = new_df[~cond]
    # Artificial Training
    y = xtrain["monthlyincome"]
    X = xtrain.drop(['monthlyincome'], axis=1)
    # Testing on missing data
    xtest = new_df[cond].drop(['monthlyincome'], axis=1)
    # RandomForest
    clf = RandomForestClassifier(n_estimators=20, max_depth=2, random_state=0, n_jobs=-1)
    model = clf.fit(X, y)
    pred = model.predict(xtest)
    new_df.loc[xtest.index.values,'monthlyincome'] = pred
    return new_df


def feature_eng(features):
    """
    Feature Engineering:
    - Creates Zip Code dummines
    - Normalizes monthly income
    - Normalizes Total balance on credit cards and personal lines of credit
    """
    # Create age dummies
    features['bins_age'] = process_data.discretize(features, 'age', range(20,120,5)).astype('category')
    features = process_data.dummify(features, 'bins_age')
    features = features.drop(['age', 'bins_age'], axis = 1)
    # Create Zipcode dummies
    features = process_data.dummify(features, 'zipcode')
    features = features.drop(['zipcode'], axis = 1)
    # Normalize income data 
    features = StandardScaler().fit_transform(features)

    return features


def do(X_data, target, model_list, hyperparameters):

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
