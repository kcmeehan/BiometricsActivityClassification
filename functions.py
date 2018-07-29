#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------
# preProcess: function to remove invalid/not useful data, remove first and last 10 seconds, etc.
#------------------------------------------------------------------------------------------------
def preProcess(filepath):
    
    # Read the data file into a pandas dataframe, choosing the timestamp column to be the df index
    df_raw = pd.read_csv(filepath, delim_whitespace=True, header=None, index_col=0)

    # Label the columns
    df_raw.columns = ['activityID',
                      'HR',
                      'hTemp',
                      'hAcc16X',
                      'hAcc16Y',
                      'hAcc16Z',
                      'hAcc06X',
                      'hAcc06Y',
                      'hAcc06Z',
                      'hGyroX',
                      'hGyroY',
                      'hGyroZ',
                      'hMagX',
                      'hMagY',
                      'hMagZ',
                      'hOrient1',
                      'hOrient2',
                      'hOrient3',
                      'hOrient4',
                      'cTemp',
                      'cAcc16X',
                      'cAcc16Y',
                      'cAcc16Z',
                      'cAcc06X',
                      'cAcc06Y',
                      'cAcc06Z',
                      'cGyroX',
                      'cGyroY',
                      'cGyroZ',
                      'cMagX',
                      'cMagY',
                      'cMagZ',
                      'cOrient1',
                      'cOrient2',
                      'cOrient3',
                      'cOrient4',
                      'aTemp',
                      'aAcc16X',
                      'aAcc16Y',
                      'aAcc16Z',
                      'aAcc06X',
                      'aAcc06Y',
                      'aAcc06Z',
                      'aGyroX',
                      'aGyroY',
                      'aGyroZ',
                      'aMagX',
                      'aMagY',
                      'aMagZ',
                      'aOrient1',
                      'aOrient2',
                      'aOrient3',
                      'aOrient4']
    df_raw.index.names = ['Timestamp']

    # Dropping columns that were invalid or not recommended for use
    df = df_raw.drop(['hAcc06X','hAcc06Y','hAcc06Z','cAcc06X','cAcc06Y','cAcc06Z','aAcc06X','aAcc06Y','aAcc06Z'], axis=1)
    df.drop(['hOrient1','hOrient2','hOrient3','hOrient4','cOrient1','cOrient2','cOrient3','cOrient4','aOrient1','aOrient2','aOrient3','aOrient4'], axis=1, inplace=True)
    df.drop(['hGyroX','hGyroY','hGyroZ','cGyroX','cGyroY','cGyroZ','aGyroX','aGyroY','aGyroZ'], axis=1, inplace=True)
    df = df[df.activityID != 0]

    # Dropping the first and last 10 seconds of the data set 
    df.drop(df.index[:1000], inplace=True)
    df.drop(df.index[-1000:], inplace=True)

    return df

#------------------------------------------------------------------------------------------------
# Creating a function to perform segmentation to prepare for feature extraction
#------------------------------------------------------------------------------------------------

def segmentation(df):

