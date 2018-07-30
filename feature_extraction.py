#feature_extraction

import numpy as np
import pandas as pd
import sys

def feat1(blocki):
    #given blocki, a T-dim array, computes mean and std
    mean=np.mean(blocki)
    std=np.std(blocki)
#    peak=np.max(blocki)
    return mean,std

def integral(series,imax=None):
    #given array with positive entries, compute integral along axis 0 up to entry i, normalized such that the integral is 1 when i reaches the end
    if imax==None:
        imax=len(series)
        tot=np.sum(series)
    return np.array([np.sum(series[:i]) for i in range(imax)])/tot

def feat3(blockixyz,ulist):
    #similar to feat2, but the power is summed over x,y,z
    T=len(blockixyz[0])
    fourierx=np.fft.rfft(blockixyz[0])
    fouriery=np.fft.rfft(blockixyz[1])
    fourierz=np.fft.rfft(blockixyz[2])
    
    power_int=integral(abs(fourierx)**2+abs(fouriery)**2+abs(fourierz)**2)
    freq_list=[np.searchsorted(power_int,u) for u in ulist]
    return np.array(freq_list)*100/T

#feature names

def feat1name(s):
    return [s+x for x in ['mean','std']]

def feat2name(s,ulist):
    return [s+'freq_'+str(x) for x in ulist]

def feat_names(ulist):
    names=feat1name('heart_rate_')
    for part in ['hand_','chest_','ankle_']:
        names+=feat1name(part+'temp_')
        for subl1 in ['a16_','gyro_','B_']:
            for subl2 in ['x_','y_','z_']:
                names+=feat1name(part+subl1+subl2)
            names+=feat2name(part+subl1,ulist)
    return names

def feat_row(block,ulist):
    #extract features from block. uses feat1 for all variables, uses feat3 for a16,gyro and B data
    heart_feat=feat1(block[:,col_dict['heart_rate']])
    all_feat=heart_feat
    for part in ['hand_','chest_','ankle_']:
        part_feat=feat1(block[:,col_dict[part+'temp']])
        for subl1 in ['a16_','gyro_','B_']:
            for subl2 in ['x','y','z']:
                part_feat=np.concatenate((part_feat,feat1(block[:,col_dict[part+subl1+subl2]])))
            nx,ny,nz=[col_dict[part+subl1+i] for i in ['x','y','z']]
            part_feat=np.concatenate((part_feat,feat3((block[:,nx],block[:,ny],block[:,nz]),ulist)))
        all_feat=np.concatenate((all_feat,part_feat))
    return all_feat

HR_sub=[(60,200),(75,193),(74,195),(68,189),(58,196),(70,194),(60,194),(60,197),(66,188),(54,189)]

def HR_norm(x,subject_n):
    HR_res0,HR_max0=HR_sub[0]
    HR_resn,HR_maxn=HR_sub[subject_n]
    x_norm=(HR_res0*(HR_maxn-x)+HR_max0*(x-HR_resn))/(HR_maxn-HR_resn)
    return x_norm

IMUlabels=['temp']+[x+y for x in ['a16_','a6_','gyro_','B_'] for y in ['x','y','z']]+['orient_'+x for x in ['1','2','3','4']]
col_labels=['time','activity_ID','heart_rate']+[x+y for x in ['hand_','chest_','ankle_'] for y in IMUlabels]
#dictionary converting activity id to activity name
activity_dict={0:'other',1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',6:'cycling',7:'nordic walking',
              9:'watching TV',10:'computer work',11:'car driving',12:'ascending stairs',13:'descending stairs',
              16:'vacuum cleaning',17:'ironing',18:'folding laundry',19:'house cleaning',20:'playing_soccer',
              24:'rope jumping'}

#sub column labels to be retained
IMUsublabels=[x+y for x in ['a16_','gyro_','B_'] for y in ['x','y','z']]
col_sublabels=['time','activity_ID','heart_rate']+[x+y for x in ['hand_','chest_','ankle_'] for y in ['temp']+IMUsublabels]


#dictionary converting column name to column index
col_dict={col_sublabels[i]:i for i in range(len(col_sublabels))}

def extract(subject_n):

    #read data for subject 101 from file, coplied mostly from cylin
    data=pd.read_csv('./PAMAP2_Dataset/Protocol/subject10'+str(subject_n)+'.dat',sep=' ',names=col_labels,header=None)

    #linear interpolate missing data
    dataint=data.interpolate(method='linear')

    #drop columns for orientation and a_6
    data_sub=pd.DataFrame(dataint,columns=col_sublabels)

    #drop activity 0
    data_nz=data_sub.loc[lambda x:x['activity_ID']!=0]

    #convert to numpy: I am not familiar with pandas enough...
    data_ar=np.array(data_nz)

    #normalize heart rate
    data_ar[:,2]=HR_norm(data_ar[:,2],subject_n)

    #split data into maximal chunks of time, with same activity ids
    x=np.arange(len(data_ar))
    split_ind=np.argwhere(data_ar[x,1]!=data_ar[x-1,1])[:,0]
    split_ind=np.append(split_ind,len(data_ar+1))
    chunks=[data_ar[split_ind[i]:split_ind[i+1]] for i in range(len(split_ind)-1)]

    #drop the first and last 10 seconds
    chunks_chopped=[x[1000:-1000] for x in chunks]

    ulist=[0.5,0.75,0.9,0.95]

    #collect data into traindata and trainlabels

    traindata=np.empty((0,98))
    trainlabels=np.empty((0))
    T=512
    stride=100
    for chunki in chunks_chopped:
        imax=(len(chunki)-T)//stride
        for i in range(imax):
            feat_rowi=feat_row(chunki[i*stride:i*stride+T],ulist)
            traindata=np.append(traindata,[feat_rowi],axis=0)
            trainlabels=np.append(trainlabels,chunki[0,1])
            
    #attach label column to data matrix.
    datalabels=np.hstack((traindata,trainlabels[:, np.newaxis]))

    np.save('data'+str(subject_n)+'.npy', (feat_names(ulist),datalabels))

    return None