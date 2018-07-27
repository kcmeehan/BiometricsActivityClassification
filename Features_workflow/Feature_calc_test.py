#!/usr/bin/env python 

#Tools to run feature calcuation on the biometric dataset

import numpy as np
import pandas as pd
import os
import glob
from FeatureCalculate import FeatureCalc


def exportColName():
    
    '''
    Create columns names
    '''
    
    handColName=['hand_temp', 'hand_acc16g_x', 'hand_acc16g_y', 'hand_acc16g_z', 'hand_acc6g_x', 'hand_acc6g_y', 'hand_acc6g_z', 
             'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z', 'hand_mag_x', 'hand_mag_y', 'hand_mag_z', 'hand_ori_0', 'hand_ori_1', 
             'hand_ori_2', 'hand_ori_3']
    chestColName=['chest_temp', 'chest_acc16g_x', 'chest_acc16g_y', 'chest_acc16g_z', 'chest_acc6g_x', 'chest_acc6g_y', 
                  'chest_acc6g_z', 'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z', 'chest_mag_x', 'chest_mag_y', 'chest_mag_z', 
                  'chest_ori_0', 'chest_ori_1', 'chest_ori_2', 'chest_ori_3']
    ankleColName=['ankle_temp', 'ankle_acc16g_x', 'ankle_acc16g_y', 'ankle_acc16g_z', 'ankle_acc6g_x', 'ankle_acc6g_y', 
                  'ankle_acc6g_z', 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z', 'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z', 
                  'ankle_ori_0', 'ankle_ori_1', 'ankle_ori_2', 'ankle_ori_3']
    return ['timestamp', 'activityID', 'heart_rate']+handColName+chestColName+ankleColName


def remove_activity_start_end(df,n=1000,dt=0.01):
    
    '''
    Remove n samples from the start and end of each activity within a subject dataframe. The Reiss paper used
    10 seconds (1000 samples)
    '''
    
    df_redacted_parts = []
        
    #Detect changes in activity ID
    activities = df['activityID']
    diffs = np.diff(activities)
    gap_indices = np.where(abs(diffs)>1e-6)[0]
    
    #Generate a list of the start and end indices of each activity
    gap_indices_list = []

    if len(gap_indices > 0):
        for i in range(len(gap_indices)):
            if i == 0:
                gap_indices_list.append((0,gap_indices[i]))
            else:
                gap_indices_list.append((gap_indices[i-1]+1,gap_indices[i]))
                
        gap_indices_list.append((gap_indices[-1]+1,len(df)))
    else:
        gap_indices_list.append((0,len(df)))
    
    #Loop though the start and end index pairs and remove the first and last n datapoints
    
    for index_pair in gap_indices_list:
        if ((index_pair[1]-n)-(index_pair[0]+n)) > 0:
            df_redacted_parts.append(df.loc[index_pair[0]+n:index_pair[1]-n])
        else:
        	print ("Duration of activity not long enough to redact data")
            #print(index_pair[1]-n,index_pair[0]+n)

    #concatinate the redacted dataframes
    redacted_df = pd.concat(df_redacted_parts)
    
    return redacted_df


def loadSubject(filename):
    
    '''
    Load a single subject from file and return a dataframe
    '''
    
    col=exportColName()
    index = int(filename.split('.')[0][-1])
    tempData = pd.read_csv(filename, sep=' ', names=col)
    tempData['subject'] = (index)*np.ones(len(tempData))
    interpData = interpolate_all(tempData)
    redacteddf = remove_activity_start_end(interpData)
    redacteddf.reset_index(inplace=True)
    return redacteddf

def loadSubject_unredacted(filename):
    
    '''
    Load a single subject from file and return a dataframe
    '''
    
    col=exportColName()
    index = int(filename.split('.')[0][-1])
    tempData = pd.read_csv(filename, sep=' ', names=col)
    tempData['subject'] = (index)*np.ones(len(tempData))
    interpData = interpolate_all(tempData)
    #redacteddf = remove_activity_start_end(interpData)
    #redacteddf.reset_index(inplace=True)
    return interpData


def interpolate_all(df):
    
    '''
    Interpolate values in the columns of a dataframe so that they have the
    same sampling rate as the other columns
    '''
    
    df.interpolate(inplace=True)
    
    return df

def loadAllSubjects(dirname):
    
    '''
    Load all subject files & return a dataframe
    '''
    
    if os.path.exists(dirname):
        dfiles = list(sorted(glob.glob("%s/*.dat" %dirname)))
    else:
        print ("Given dirname %s not found" %dirname)
        
    col=exportColName()
    dfs = []
    
    for i in range(len(dfiles)):
        filename=dfiles[i]
        df = loadSubject(filename)
        dfs.append(df)
    
    allData = pd.concat(dfs)
    allData.dropna(inplace=True)
    allData.reset_index(drop=True,inplace=True)
    
    return allData

def select_data_to_process(df):

    '''
    Remove unwanted columns and activity 0
    '''

    columns_to_keep = ['timestamp', 'activityID', 'heart_rate','hand_temp', 'hand_acc16g_x', 'hand_acc16g_y', 
                   'hand_acc16g_z','hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z', 
                   'hand_mag_x', 'hand_mag_y', 'hand_mag_z','chest_temp', 'chest_acc16g_x', 
                   'chest_acc16g_y', 'chest_acc16g_z','chest_gyro_x', 'chest_gyro_y', 
                   'chest_gyro_z', 'chest_mag_x', 'chest_mag_y', 'chest_mag_z','ankle_temp', 
                   'ankle_acc16g_x', 'ankle_acc16g_y', 'ankle_acc16g_z','ankle_gyro_x', 
                   'ankle_gyro_y', 'ankle_gyro_z', 'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z','subject']

    df = df[columns_to_keep]
    df = df[df['activityID'] != 0]

    return df

def calculate_features(all_data,dt=0.01,sliding_window_length=512,sliding_window_offset=100):

	'''
	Load a dataframe called all_data and calculate features
	'''

    #Estimate the number of rows needed in the matrix where we will store all the data. This will be
    #an over-estimate, but we can remove the unfilled rows later
	nslices_estimate = int(np.floor((len(all_data)-sliding_window_length)/sliding_window_offset))

	#This part will need editing with the names of the new features that we calculate
	features = ['mean','median','std','peak']

    # We don't want to calculate mean features for the timestamp, subject or activity columns, but
    # we do want them to appear in the final feature matrix

	cols = all_data.columns

	feature_cols = []
	for featurename in features:
	    for name in cols:
	        newname = '%s_%s' %(name,featurename)
	        feature_cols.append(newname)

	ncols = len(feature_cols)

	#Define featurecalc object
	#This is a class to which new timeseries can be loaded, and features calculated 
	f = FeatureCalc(ncols)

	#We will fill the feature matrix and then turn it into a dataframe because we 
	#have already generated the column names. If there are extra nans we will just remove them
	#at the end
	feature_matrix = np.full((nslices_estimate,ncols),np.nan)

	rowcount=0

	for subjectID in all_data['subject'].unique():
	    
	    print('\n-------------------------------')
	    print('Subject %s' %subjectID)
	    
	    subjectDF = all_data[all_data['subject'] == subjectID]
	    subjectDF.reset_index(inplace=True,drop=True)
	      
	    #Detect changes in activity ID
	    activities = subjectDF['activityID']
	    diffs = np.diff(activities)
	    gap_indices = np.where(abs(diffs)>1e-6)[0]
	    
	    #Generate a list of the start and end indices of each activity
	    gap_indices_list = []

	    if len(gap_indices > 0):
	        for i in range(len(gap_indices)):
	            if i == 0:
	                gap_indices_list.append((0,gap_indices[i]))
	            else:
	                gap_indices_list.append((gap_indices[i-1]+1,gap_indices[i]))
	                
	        gap_indices_list.append((gap_indices[-1]+1,len(subjectDF)))
	    else:
	        gap_indices_list.append((0,len(subjectDF)))
	            
	    print(gap_indices_list)

	    for index_pair in gap_indices_list:

	        constant_activity_slice = subjectDF.loc[index_pair[0]:index_pair[1]]
	        constant_activity_slice.reset_index(inplace=True,drop=True)
	        #print(constant_dt_slice['timestamp'].is_monotonic)
	        
	        print('Activity %f' %np.mean(constant_activity_slice['activityID']))

	        #Check if the slice that we've made has enough samples to calculate features

	        if len(constant_activity_slice) < sliding_window_length:
	            print("Not enough points to create slice of length %i" %sliding_window_length)

	        else:

	            #Move though the slice in this number of units. We will lose some data
	            #at the end of each constant_dt_slice dataframe because we must take an integer
	            #number of steps 

	            nslices = int(np.floor((len(constant_activity_slice)-sliding_window_length)/sliding_window_offset))

	            print("Length of slice in samples: %i" %len(constant_activity_slice))
	            print("Number of feature calculations to be done: %i" %nslices)

	            t1 = 0
	            for j in range(nslices):
	                t2 = t1 + sliding_window_length - 1
	                #print(t1,t2)
	                
	                feature_slice = constant_activity_slice.loc[t1:t2]
	                    
	                if (abs(np.mean(np.diff(feature_slice['timestamp']))) - dt > 1e-6):
	                    print("Error in slicing gap indices! Data chunk is not sampled at constant dt!")
	                
	                else:

	                    #This is the dataframe on which we will calculate statistical features
	                    #feature_slice.drop(['timestamp'],axis=1,inplace=True)

	                    #Feature calculation step - the feature calculation happens inside FeatureCalculate.py

	                    f.load_new_ts(feature_slice)
	                    data = f.calculate_features()

	                    feature_matrix[rowcount,:] = data
	                    rowcount+=1
	                

	                t1 = t1 + sliding_window_offset

	feature_df = pd.DataFrame(feature_matrix,columns=feature_cols)

	#Drop unwanted columns from the feature df (all the features of the three dropcols, aside from the mean, which
	#may be useful)

	unwanted_features_head = features[1:]
	dropcols = ['timestamp','activityID','subject']
	unwanted_cols = []

	for element in dropcols:
		for fname in unwanted_features_head:
			colname = '%s_%s' %(element,fname)
			unwanted_cols.append(colname)

	feature_df.drop(unwanted_cols,inplace=True,axis=1)



	return feature_df

