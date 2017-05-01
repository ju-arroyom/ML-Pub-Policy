import os
import pandas as pd

def read_data(raw_file):
    '''
    Read raw data file, check file file_extension
    and decide the best way to read the file.
    The function currently works for csv files, but
    it may extended.
    Inputs: raw data file
    Returns : data structure for manipulation
    '''
    filename, file_extension = os.path.splitext(raw_file)
    if file_extension.endswith('.csv'):
        df =  pd.read_csv(raw_file, header=0)
        # Lower case names for columns
        df.columns = map(str.lower, df.columns)

    return df