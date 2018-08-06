#!/usr/bin/env python

#Functions to aid in the determination of optimal hyperparameters and features from the 
#biometric activity dataset

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from GeneticAlgorithm import GeneticAlgorithm
import time
import pandas as pd

def test_model_initial(model,X,Y,test_parameters,hold_out_fraction=0.3):
    
    '''
    Explore parameter space and report the best model and feature selection

    This uses SelectFromModel to reduce the number of features based on their importances and
    GridSearchCV to search for the best parameters

    Input:
    model is an sklearn classifier object
    X is the input feature dataframe
    Y is the target vector
    test_parameters is a dictionary to be fed into GridSearchCV
    hold_out_fraction is the fraction of the dataset to use as a test once the hyperparameters have 
    been tuned

    Output:
    X_new: dataframe of the optimal features
    best_classifier: the model with the best hyperparameters
    '''
    
    #Split into training and hold-out sets. We are going to do our search for optimal 
    #hyperparameters on the training set and preserve the holdout set for testing 
    #later
    
    #Label encoder
    le = LabelEncoder()
    labels = le.fit_transform(Y)
    
    #Scaler
    Scaler = StandardScaler()
    X_scaled = Scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,labels,test_size=hold_out_fraction)
    
    #Random forest classifier
    CF = model
    #Feature selector
    FS = SelectFromModel(CF, threshold='mean',prefit=False)
    
    #Pipeline
    CF_pipeline = Pipeline([('select', FS), ('classify', CF)])
    
    #Search parameter space
    grid_search = GridSearchCV(CF_pipeline, test_parameters, verbose=1, cv=5, n_jobs=4)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in CF_pipeline.steps])
    print("parameters:")
    print(test_parameters)
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(test_parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    #Test on the hold_out set
    p = grid_search.best_estimator_
    best_classifier = p.named_steps['classify']
    
 
    #Determine score on hold-out dataset using the selected parameters
    hold_out_score = best_classifier.score(p.named_steps['select'].transform(X_test)
                                           ,y_test)
    
    print("Hold out score: %0.3f" %hold_out_score)
    
    X_new = p.named_steps['select'].transform(X)
    support = p.named_steps['select'].get_support()
    X_new_cols = [X.columns[i] for i in range(len(support)) if support[i] == True]
    
    X_new = pd.DataFrame(X_new,columns=X_new_cols)
    
    return X_new, best_classifier


def Run_GA(X,Y,classifier):

	'''
	Run generic algorithm to determine the optimal feature selection. Run this on the original datasets 

	Inputs:
	X: dataframe containing all features
	Y: target variable
	classifier: sklean classifier object with chosen hyperparameters

	Outputs:
	GeneticAlgorithm object 
	'''

	GA = GeneticAlgorithm(X,Y,classifier,Niter=100 ,njobs=4)
	GA.fit()

	return GA


def LOSO(full_df,classifier,hold_out_fraction=0.3):
    
    '''
    Do Leave-One-Subject-Out cross validation on the full dataset

    This could be carried out once the optimal hyperparameters have been chosen using the function above

    Inputs:
    full_df dataframe containing columns of subject ID and activity ID
    classifier: SKlearn object with chosen hyperparameters
    hold_out_fraction: fraction of total data to reserve for testing 

    Output:
    Scores for each hold-out subject
    '''
    
    #Do a test-train split. This is just to get the dataframe X_train_full, which contains all the information
    #we need to do the cross validation
    
    #Once we've turned to hyperparameters, we can test the model on X_test_holdout and Y_test_holdout

    Y = full_df['activityID']

    #Label encoder
    le = LabelEncoder()
    labels = le.fit_transform(Y)
    
    X_train_full, X_test_holdout, Y_train_full, Y_test_holdout = train_test_split(full_df,labels,test_size=hold_out_fraction)

    
    ######
    #Do cross validation 
    ######
    
    subject_scores = {}
    
    for subjectID in X_train_full['subjectID'].unique():
        
        print("Holding out subject %i" %subjectID)
        
        #Build test dataset - select just that subject and their associated activities
        X_test = X_train_full[X_train_full['subjectID']==subjectID]
        Y_test = X_test['activityID'].copy()
        X_test.drop(['activityID','subjectID'],inplace=True,axis=1)
        
        #Build training dataset - select all but that subject and their associated activities
        X_train = X_train_full[X_train_full['subjectID']!=subjectID]
        Y_train = X_train['activityID'].copy()
        X_train.drop(['activityID','subjectID'],inplace=True,axis=1)
        
        classifier.fit(X_train,Y_train)
        
        score = classifier.score(X_test,Y_test)
        subject_scores[str(subjectID)] = score 
        
    return subject_scores

