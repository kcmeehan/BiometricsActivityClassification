import pandas

### Create column names ###
def exportColName():
    handColName=['hand_temp', 'hand_acc16g_x', 'hand_acc16g_y', 'hand_acc16g_z', 'hand_acc6g_x', 'hand_acc6g_y', 'hand_acc6g_z', 
                 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z', 'hand_mag_x', 'hand_mag_y', 'hand_mag_z', 'hand_ori_0', 'hand_ori_1', 
                 'hand_ori_2', 'hand_ori_3']
    chestColName=['chest_temp', 'chest_acc16g_x', 'chest_acc16g_y', 'chest_acc16g_z', 'chest_acc6g_x', 'chest_acc6g_y', 'chest_acc6g_z', 
                  'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z', 'chest_mag_x', 'chest_mag_y', 'chest_mag_z', 'chest_ori_0', 'chest_ori_1', 
                  'chest_ori_2', 'chest_ori_3']
    ankleColName=['ankle_temp', 'ankle_acc16g_x', 'ankle_acc16g_y', 'ankle_acc16g_z', 'ankle_acc6g_x', 'ankle_acc6g_y', 'ankle_acc6g_z', 
                  'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z', 'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z', 'ankle_ori_0', 'ankle_ori_1', 
                  'ankle_ori_2', 'ankle_ori_3']
    return ['timestamp', 'activityID', 'heart_rate']+handColName+chestColName+ankleColName



### Load a single subject file & Return a dataframe ###
# subjectIndex=1,2,3...9
def loadSubject(subjectIndex):
    filename="PAMAP2_Dataset/Protocol/subject10"+str(subjectIndex)+".dat"
    col=exportColName()
    return pandas.read_csv(filename, sep=' ', header=None, names=col)


### Load all subject files & Return a dataframe ###
def loadAllSubjects():
    allData=loadSubject(1)
    col=exportColName()
    for i in range(2, 10):
        filename="PAMAP2_Dataset/Protocol/subject10"+str(i)+".dat"
        tempData=pandas.read_csv(filename, sep=' ', header=None, names=col)
        allData=allData.append(tempData)
          
    allData=allData.reset_index(drop=True)
    return allData


