import numpy as np
import pandas as pd
import FeatureCalculate as FC

def HR_norm(x,HR_rest,HR_max):
    """
    Normalize the heart rates of subject such that the normalized resting/maximum heart rate is HR_res0/HR_max0
    type x: float --input heart rate
    type HR_rest,HR_max: float --the resting and maximum heartrates of the subject, can be found in subjectInformation.pdf
    return type: float --normalized heart rate. Resting heart rate 
    """
    HR_res0,HR_max0=HR_lim[0]
    x_norm=(HR_res0*(HR_max-x)+HR_max0*(x-HR_rest))/(HR_max-HR_rest)
    return x_norm

#Heart rate limits for subject copied from subjectInformation.pdf. the 0th element is the standardized limit
HR_lim=[(60,200),(75,193),(74,195),(68,189),(58,196),(70,194),(60,194),(60,197),(66,188),(54,189)]

#labels of columns in raw data. 54 columns in total
IMUlabels=[x+y for x in ['acc16g_','acc6g_','gyro_','mag_'] for y in ['x','y','z']]+['ori_'+x for x in ['0','1','2','3']]
col_labels=['timestamp','activityID','heart_rate']+[x+y for x in ['hand_','chest_','ankle_'] for y in ['temp']+IMUlabels]

#labels of columns in preprocessed data. 33 columns in total
IMUsublabels=[x+y for x in ['acc16g_','gyro_','mag_'] for y in ['x','y','z']]
col_sublabels=['timestamp','activityID','heart_rate']+[x+y for x in ['hand_','chest_','ankle_'] for y in ['temp']+IMUsublabels]

#Converting column name to preprocessed column index. eg. col_dict['heart_rate'] returns 2
col_dict={col_sublabels[i]:i for i in range(len(col_sublabels))}

#Converting activity id to activity name
activity_dict={0:'other',1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',6:'cycling',7:'nordic walking',
              9:'watching TV',10:'computer work',11:'car driving',12:'ascending stairs',13:'descending stairs',
              16:'vacuum cleaning',17:'ironing',18:'folding laundry',19:'house cleaning',20:'playing_soccer',
              24:'rope jumping'}

class dataprocess():
    def __init__(self,subj_filename,T=512,stride=512,redact=1000):
        """
        type subject_filename: str --the filepath of .dat data file for one subject
        T -- number of samples on which to calculate features
        stride -- gap between chunks of data onwhich features are calculated
        redact -- number of samples to remove from the start and end of each activity 
        
        """
        self.subj_filename=subj_filename
        
        #Get the resting and maximum heartrates for that subject
        subject_index = int(self.subj_filename.split('/')[-1].split('.')[0][-1])
        self.HR_rest = HR_lim[subject_index][0]
        self.HR_max = HR_lim[subject_index][1]
        
        self.feat_labels=None
        self.chunks=self.preprocess(redact)
        self.fc = FC.FeatureCalc()
        self.data_segmented=self.segmentation(self.chunks,T,stride)
        self.df=pd.DataFrame(self.data_segmented,columns=self.feat_labels)
        
        
    def preprocess(self,redact):
        """
        return type : List[arrays] --list of proprocessed continuous segments (each is several minutes long) of data, within which the
        subject performs only one activity, the arrays have 33 columns (including timestamp at column 0, activity_ID at column 1, and heartrate at column 2)
        """
        
        #generate dataframe from the raw data
        data=pd.read_csv(self.subj_filename,sep=' ',names=col_labels,header=None)

        #linear interpolate missing data
        dataint=data.interpolate(method='linear')

        #drop columns for orientation and acc6g
        data_sub=pd.DataFrame(dataint,columns=col_sublabels)

        #convert to array
        data_ar=np.array(data_sub)

        #normalize heart rate
        data_ar[:,2]=HR_norm(data_ar[:,2],self.HR_rest,self.HR_max)

        #computes timestamp indices where the activity changes, including 0 and l
        l=len(data_ar)
        r=np.arange(l-1)+1
        split_ind=r[data_ar[r,1]!=data_ar[r-1,1]]
        split_ind=np.concatenate(([0],split_ind,[l]))

        #chop data into chunks of continuous time blocks with the same activity, also remove activity zero
        chunks=[data_ar[split_ind[i]:split_ind[i+1]] for i in range(len(split_ind)-1) if data_ar[split_ind[i],1]!=0]
        
        #drop the first and last n samples. Only keep redacted samples that 
        #are of sufficient length
        
        chunks=[x[redact:-(redact+1)] for x in chunks if len(x) > (2*redact)]

        return chunks
        
    def segmentation(self,chunks,T=512,stride=512):
        """
        type chunks:List[array] --continuous time blocks of data, each array should have 33 columns
        return type: array --each row corresponds to features calculated on a segment of length T. Features are extracted by
        feature_extraction. Consecutive rows are calculated on segments which are one stride apart.
        """
        data_segmented=np.empty(0)
        for chunk in chunks:
            imax=(len(chunk)-T)//stride
            for i in range(imax+1):
                arr=self.feature_extraction(chunk[i*stride:i*stride+T])
                data_segmented=np.vstack((data_segmented,arr)) if data_segmented.size else arr
        return data_segmented

    def feature_extraction(self,segment):
        """
        type segment:array with shape (512,33)
        return type: array with shape (ncol)

        Performs feature_extraction using the FeatureCalc class from Robert.
        Feature labels are imported from feat_labels attribute from a FeatureCalc object.
        """
        
        segment_df=pd.DataFrame(segment,columns=col_sublabels)
        self.fc.load_new_ts(segment_df)
        if self.feat_labels==None:
            self.feat_labels=self.fc.feat_labels
        arr=self.fc.calculate_features()
        return arr