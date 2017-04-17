import read_files
import pandas as pd
import numpy as np


def des_num_dep (data):
    '''
    Creates data frame with cumsum and percentage of 
    dependents by category
    Input: pandas data frame object
    Returns: new df with descriptive stats
    '''
    data = data['numberofdependents'].value_counts().to_frame()
    data['cumsum'] = data['numberofdependents'].cumsum()
    total = data['numberofdependents'].sum()
    data['percentage'] = (data['cumsum'] / total)*100 
    
    return data




def impute_val_to_column(data, col, method, l_bound = 0, u_bound = 1, freq_list = []):
    '''
    Given a list of specific data columns, impute missing
    data of those columns with the column's mean, median, or mode.
    This function imputes specific columns, for imputing all
    columns of the dataset that have missing data, use
    impute_missing_all.
    '''

    if method == 'median':
        data[col] = data[col].fillna(data[col].median())
    elif method == 'mode':
        data[col] = data[col].fillna(int(data[col].mode()[0]))
    elif method == 'random':
        data[col] = data[col].fillna(np.random.choice(np.arange(l_bound, u_bound), p= freq_list))
    else:
        data[col] = data[col].fillna(data[col].mean())




def discretize(data, column, bins):
    '''
    Given a continuous variable, create a new column in the dataframe
    that represents the bin in which the continuous variable falls into.
    If verbose is True, print the value counts of each bin.
    Returns the name of the new column to include programmatically in list of features
    '''
    new_col = 'bins_' + str(column)

    data[new_col] = pd.cut(data[column], bins=bins)

    return data[new_col]