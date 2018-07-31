#!/usr/bin/env python

#Class to calculate time series features for the Biometric data classification project

import numpy as np
import pandas as pd

class FeatureCalc:

    def __init__(self):

        '''

        Takes in a pandas dataframe consisting of columns whose features we want to calculate

        self.ncols is the total number of columns expected to be in the final feature dataframe. Its used 
        to preallocate a numpy array that gets filled with the feature values.

        '''

        #self.ncols = ncols

    def load_new_ts(self,input_df):

        '''
        Load a new data chunk for analysis 
        '''

        self.indf = input_df

        #Feature columns for three component analysis
        self.hand_acc = self.indf[['hand_acc16g_x','hand_acc16g_y','hand_acc16g_z']]
        self.hand_gyro = self.indf[['hand_gyro_x','hand_gyro_y','hand_gyro_z']]
        #self.hand_mag = self.indf[['hand_mag_x','hand_mag_y','hand_mag_z']]
        self.chest_acc = self.indf[['chest_acc16g_x','chest_acc16g_y','chest_acc16g_z']]
        self.chest_gyro = self.indf[['chest_gyro_x','chest_gyro_y','chest_gyro_z']]
        #self.chest_mag = self.indf[['chest_mag_x','chest_mag_y','chest_mag_z']]
        self.ankle_acc = self.indf[['ankle_acc16g_x','ankle_acc16g_y','ankle_acc16g_z']]
        self.ankle_gyro = self.indf[['ankle_gyro_x','ankle_gyro_y','ankle_gyro_z']]
        #self.ankle_mag = self.indf[['ankle_mag_x','ankle_mag_y','ankle_mag_z']]

        #feature names
        self.feat_labels=[x+y for y in ['_mean','_median','_std','_peak'] for x in self.indf.columns]

    def calculate_features(self):

        '''
        The calculation of features must be in the same order as assumed by the
        input matrix
        '''

        #These are features that can be calculate quickly on all columns of the dataframe using 
        #pandas. There may be other features that won't work like this, but we can calculate them 
        #indiivdually and then append them to this arr object

        features = [self.mean,self.median,self.std,self.peak]

        arr = np.empty(0)

        for featurefunc in features:
            values = featurefunc()
            arr = np.concatenate((arr,values))
        return arr

    #########################
    # Feature calculators
    #########################

    def mean(self):
        
        '''
        Return mean of all the columns in the df
        '''

        return self.indf.mean(skipna=False).values

    def median(self):
        
        '''
        Return median of all the columns in the df
        '''

        return self.indf.median(skipna=False).values

    def std(self):
        
        '''
        Return std of all the columns in the df
        '''

        return self.indf.std(skipna=False).values

    def peak(self):
        
        '''
        Return peak of all the columns in the df
        '''

        return self.indf.abs().max().values
    
    def kurtotis(self):
        
        '''
        Return kurtosis of all the columns in the df
        '''
        
        return self.indf.kurt(skipna=False).values

