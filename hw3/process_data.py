
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats as ttest

def percentage_missing(data):
    '''
    Computes the percentage of missing values by
    column in the dataframe
    Input: dataframe
    Return: prints percentage of missing values by column
    '''
    for c in data.columns:
        if data[c].count() < len(data):
            missing_perc = ((len(data) - data[c].count()) / float(len(data))) * 100.0
            print("%.1f%% missing from: Column %s" %(missing_perc, c))


def frequency_plots(data, col_name):
    '''
    Creates a plot to inspect the distribution of a variable
    Inputs:
            data - pandas dataframe
            col_name - column name
    Returns: 
            plot
    '''
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,7))
    plt.rcParams['legend.fontsize'] = 20

    data[col_name].plot(kind = "hist", alpha = 0.2, bins = 20, color ='r', ax = ax1); 
    ax1.set_title('Distribution of Number of Dependents');

    sns.boxplot(data[col_name], ax = ax2, palette="Set3"); 
    ax2.set_title('Boxplot of Number of Dependents')

    plt.tight_layout()

def plot_income_distribution(data, col_name):
    '''
    Create subplots for income distribution

    Input: data - pandas dataframe
           col_name - column name
    Returns:
            Income distribution plot
    '''

    fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2,2, figsize=(10,10))
    # Overall Income Distribution
    sns.boxplot(data[col_name], ax = ax1); ax1.set_title('Overall income distribution')
    # Up to 80th percentile
    data.monthlyincome[(data[col_name]<=54166)].plot(kind = "hist", alpha = 0.2, color ='blue', bins= 20, ax = ax2); 
    ax2.set_title('Monthly Income Dist. up to 80th percentile');


    # Monthly income above $54,166 and below $1,000,000
    data.monthlyincome[(data[col_name]> 54166) & (data[col_name]<=1000000)].plot(kind = "hist", alpha = 0.2, color ='r', bins= 20, ax = ax3); 
    ax3.set_title('Monthly Income Dist. > 80th percentile < $1,000,000');
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0));
    ax3.set_xlabel('Income in Sci Notation')

    # Income distribution above $1,000,000
    data.monthlyincome[(data[col_name]>1000000)].plot(kind = "hist", alpha = 0.2, color ='black', bins= 20, ax = ax4); 
    ax4.set_title('Monthly Income Dist. above $1,000,000');
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(0,0));
    ax4.set_xlabel('Income in Sci Notation')

    plt.tight_layout()


def des_by_category (data, label):
    '''
    Creates data frame with cumsum and percentage of 
    dependents by category

    Input: pandas data frame object
    Returns: new df with descriptive stats
    '''
    data = data[label].value_counts().to_frame()
    data['Cumulative_Sum'] = data[label].cumsum()
    data = data.rename(columns={label: 'Count_By_Group'})
    total = data['Count_By_Group'].sum()
    data['Percentage_By_Group'] = (data['Cumulative_Sum'] / total)*100 
    data.index.name = label
    return data



def impute_val_to_column(data, col, method, l_bound = 0, u_bound = 1, freq_list = []):
    '''
    Given a list of specific data columns, impute missing
    data of those columns with the column's mean, median, or mode.
    This function imputes specific columns, for imputing all
    columns of the dataset that have missing data, use
    impute_missing_all.

    Input: data - pandas dataframe
           col - column name
           method - median, mode, random
           l_bound - lower bound for range in random imputation (default = 0)
           u_bound - upper bound for range in random imputation (default = 0)
           freq_list - list containing the empirical frequency of values in the data 
                     (defalt = [])
    Returns:
            dataframe with imputed values

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
    
    Inputs: data - pandas dataframe
            column - column name
            bins - a range to discretize the data

    Returns: the column with the buckets
    '''

    new_col = 'bins_' + str(column)

    data[new_col] = pd.cut(data[column], bins=bins)

    return data[new_col]



def dummify(data, col_name):
    '''
    Given a pandas dataframe and a column name
    creates dummy variables indicating whether the
    category was present or not.

    Inputs: data - pandas dataframe
            col_name - column name

    Returns: modified dataframe with dummy variables
    '''
    dummy = pd.get_dummies(data[col_name])
    df_add_dummy = pd.concat([data, dummy], axis =1 )
    return df_add_dummy

def visualize_buckets(data, f1, age_bin, income_bin):
    '''
    Plot features by buckets
    Inputs: data - pandas dataframe
            f1-f3  data features
            age_bin column
            income_bin column
    Returns:
            figure with 4 subplots
    '''
    fig, ((ax1, ax2))= plt.subplots(1,2, figsize=(30,20))

    data[[f1, age_bin]].groupby([age_bin]).mean().plot.area(alpha = 0.2, ax = ax1, color = 'r');
    ax1.set_title('Serious Financial Distress by Age');


    data[[f1, income_bin]].groupby([income_bin]).mean().plot.area(alpha = 0.2, ax = ax2, color = 'purple');
    ax2.set_title('Serious Financial Distress by Monthly Income');


    fig.tight_layout()


def plot_corr_matrix(data):
    '''
    Heatmap of correlation matrix
    Inputs: dataframe
    Returns: Heatmap
            (Green + corr. Red - corr.)
    '''
    sns.set(font_scale=1.4)#for label size
    ax = plt.axes()
    sns.heatmap(data.corr(), square=True, cmap='Blues')
    ax.set_title('Correlation Matrix')
    
    
def generate_test_stats(data, varlst):
    """
    Generate summary stats for test
    """
    d = {}
    for var in varlst:
        if var != "dummy":
            cond1 = data[data["dummy"] == 1][var]
            cond0 = data[data["dummy"] == 0][var]
            # Statistics for group 1
            mu1 = np.nanmean(np.asarray(cond1))
            s1  = np.nanstd(np.asarray(cond1)) 
            n1  = cond1.notnull().sum()
            # Statistics for group 0
            mu2 = np.nanmean(np.asarray(cond0))
            s2  = np.nanstd(np.asarray(cond0)) 
            n2  = cond0.notnull().sum()
            # Fill dictionary
            d[var] = (mu1,s1,n1,mu2,s2,n2)
    return d


def test_means(dictionary):
    """
    Return dictionary with p-values from ttest
    """
    test = {}
    for key, val in dictionary.items():
        mu1, s1, n1, mu2, s2, n2 = val
        test[key] = ttest(mu1, s1, n1, mu2, s2, n2).pvalue
    return test