#!/usr/bin/env python
#Tools to plot feature characteritics 

import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_distributions(data_type,features_to_plot,activities_to_plot,data=None,activity_dict=False):
    
    '''
    
    Takes a data type e.g 'heart_rate', a list of features to plot e.g. 
    ['mean', 'std', 'median'] and a list of activities to plot e.g. [1,2,3]
    and constructs a series of histrograms showing the the distribution of 
    these features.
    
    data is a dataframe containing the features
    
    Note that the elements in these lists must correspond exactly with the IDs and
    features available in the provided dataframe
    
    '''
    
    plt.style.use('ggplot')
    
    fig = plt.figure(figsize=(20,5))
    
    if activity_dict == False:
    
        activity_dict={0:'other',1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',6:'cycling',7:'nordic walking',
                  9:'watching TV',10:'computer work',11:'car driving',12:'ascending stairs',13:'descending stairs',
                  16:'vacuum cleaning',17:'ironing',18:'folding laundry',19:'house cleaning',20:'playing_soccer',
                  24:'rope jumping'}
    
    number_of_features = len(features_to_plot)
    axs = []
    for i in range(number_of_features):
        plotnp = '1%i%i' %(number_of_features,i+1)
        ax = fig.add_subplot(plotnp)
        axs.append(ax)


    for activityID in activities_to_plot:

        activitydf = data[data['activityID_mean']==activityID]

        j = 0
        for featureID in features_to_plot:

            colname = '%s_%s' %(data_type,featureID)
            sns.distplot(activitydf[colname],norm_hist=True,ax=axs[j],kde=True,bins=50,label=activity_dict[activityID])
            j+=1

        axs[0].legend()

        
    fig.suptitle('Distribution of feature values for %s' %data_type)
    
    return fig


