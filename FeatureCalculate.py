#!/usr/bin/env python

#Class to calculate time series features for the Biometric data classification project

import numpy as np
import pandas as pd
from scipy import stats, integrate, fft, signal


class FeatureCalc:

    def __init__(self):

        '''

        Takes in a pandas dataframe consisting of columns whose features we want to calculate

        self.ncols is the total number of columns expected to be in the final feature dataframe. Its used 
        to preallocate a numpy array that gets filled with the feature values.

        '''
    def load_new_ts(self,input_df):

        '''
        Load a new data chunk for analysis 
        '''

        self.indf = input_df

        #Feature columns for three component analysis
        self.hand_acc = self.indf[['hand_acc16g_x','hand_acc16g_y','hand_acc16g_z']]
        self.hand_gyro = self.indf[['hand_gyro_x','hand_gyro_y','hand_gyro_z']]
        self.chest_acc = self.indf[['chest_acc16g_x','chest_acc16g_y','chest_acc16g_z']]
        self.chest_gyro = self.indf[['chest_gyro_x','chest_gyro_y','chest_gyro_z']]
        self.ankle_acc = self.indf[['ankle_acc16g_x','ankle_acc16g_y','ankle_acc16g_z']]
        self.ankle_gyro = self.indf[['ankle_gyro_x','ankle_gyro_y','ankle_gyro_z']]

        #feature names
        self.feat_labels=[x+y for y in ['_mean','_median','_std','_peak','_kurtosis'] for x in self.indf.columns[2:]]+[x+y+z for y in ['_acc','_gyro'] for x in ['hand','chest','ankle'] for z in ['_vsum_welch','_vsum_sp_entropy','_spectrum_energy','_power_ratio','_XcY','_XcZ','_YcZ']]+['activityID']

        
    def calculate_features(self):

        '''
        The calculation of features must be in the same order as assumed by the
        input matrix
        '''

        #These are features that can be calculate quickly on all columns of the dataframe using 
        #pandas. There may be other features that won't work like this, but we can calculate them 
        #indiivdually and then append them to this arr object

        features = [self.mean,self.median,self.std,self.peak,self.kurtotis]

        arr = np.empty(0)

        for featurefunc in features:
            values = featurefunc()
            arr = np.concatenate((arr,values))
            
        #add extra features that need to be calculated on a subset of the columns
            
        for df in [self.hand_acc,self.chest_acc,self.ankle_acc,self.hand_gyro,self.chest_gyro,self.ankle_gyro]:
            
            #Calculate these features on the vector sum of the three component data. This
            #prevents the total number of features from becoming really large
            X = df.iloc[:,0]
            Y = df.iloc[:,1]
            Z = df.iloc[:,2]
            
            vector_sum = np.sqrt(np.square(X)+np.square(Y)+np.square(Z))
            
            #Features
            welch = self.peak_welch(vector_sum)
            spectrum_energy = self.spectrum_energy(X,Y,Z)
            spectral_entropy = self.spectral_entropy(vector_sum)
            power_ratio = self.power_ratio(vector_sum,[0,2.75],[0,5])
            XcY = self.correlation(X,Y)
            XcZ = self.correlation(X,Z)
            YcZ = self.correlation(Y,Z)
            
            arr = np.concatenate((arr,np.array([welch,spectral_entropy,spectrum_energy,power_ratio,XcY,XcZ,YcZ])))
        
            
        arr=np.append(arr,self.indf.activityID[0])
        
        return arr

    #########################
    # Feature calculators
    #########################
    
    #### 
    #Time feature calculators that operate on all columns
    ####

    def mean(self):
        
        '''
        Return mean of all the columns in the df
        '''

        return self.indf.mean(skipna=False).values[2:]

    def median(self):
        
        '''
        Return median of all the columns in the df
        '''

        return self.indf.median(skipna=False).values[2:]

    def std(self):
        
        '''
        Return std of all the columns in the df
        '''

        return self.indf.std(skipna=False).values[2:]

    def peak(self):
        
        '''
        Return peak of all the columns in the df
        '''

        return self.indf.abs().max().values[2:]
    
    def kurtotis(self):
        
        '''
        Return kurtosis of all the columns in the df
        '''
        
        return self.indf.kurt(skipna=False).values[2:]
    
    
    #### 
    #Time feature calculators that operate on specific columns
    ####
    
    def correlation(self,X,Y):
        
        '''
        Return the correlation coefficient between a pair of columns
        '''
        
        return stats.pearsonr(X,Y)[0]

    #### 
    #Frequency feature calculators that operate on specific columns
    ####
    
    def peak_welch(self,X):
        
        '''
        Return frequency corresponding to the peak of the Welch peridogram
        '''
        
        freqs, pspec = signal.welch(X,nperseg=256,fs=100,scaling='spectrum')
        return freqs[np.argmax(pspec)]

    def spectrum_energy(self,X,Y,Z):
        
        '''
        Return energy associated with power spectrum 
        '''
        
        N = len(X)
        fft_X = fft(X)
        fft_Y = fft(Y)
        fft_Z = fft(Z)
        energy_x = np.sum(np.abs(fft_X[1:int(N/2)]))/(N/2)
        energy_y = np.sum(np.abs(fft_Y[1:int(N/2)]))/(N/2)
        energy_z = np.sum(np.abs(fft_Z[1:int(N/2)]))/(N/2)
        energy_mean = np.mean([energy_x,energy_y,energy_z])
        
        return energy_mean
    
    def spectral_entropy(self,X):
        
        '''
        Return the spectral entropy of a signal
        Details from https://www.mathworks.com/help/signal/ref/pentropy.html
        '''
        
        freqs, pspec = signal.welch(X,nperseg=256,fs=100,scaling='spectrum')
        prob = pspec/np.sum(pspec)
        return -1*np.sum(np.multiply(prob,np.log(prob)), axis=0)
    
    def power_ratio(self,X,band1,band2):
    
        '''
        Return the power ratio between bands 1 and 2 by integrating the spectrogram
        '''
        
        freqs, pspec = signal.welch(X,nperseg=256,fs=100,scaling='spectrum')

        i1 = np.argmin(abs(freqs - band1[0]))
        i2 = np.argmin(abs(freqs - band1[1]))

        i3 = np.argmin(abs(freqs - band2[0]))
        i4 = np.argmin(abs(freqs - band2[1]))

        pv1 = integrate.simps(pspec[i1:i2],freqs[i1:i2])
        pv2 = integrate.simps(pspec[i3:i4],freqs[i3:i4])

        return (pv1/pv2)

            

