from surprise import Dataset,Reader 
from surprise import SVD,KNNBaseline
from surprise.model_selection import KFold, cross_validate
from surprise.model_selection import GridSearchCV,RandomizedSearchCV
import numpy as np
import pandas as pd
import time
import os

'''Loading the datasets of the ratings'''

#load dataset1
file_path = os.path.expanduser('./ratings_1.csv')
print("Loading Dataset1...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 5], skip_lines=1)
data1 = Dataset.load_from_file(file_path, reader=reader)
print("Done.")

#load dataset2
file_path = os.path.expanduser('./ratings_2.csv')
print("Loading Dataset2...")
reader2 = Reader(line_format='user item rating', sep=',', rating_scale=[1, 10], skip_lines=1)
data2 = Dataset.load_from_file(file_path, reader=reader2)
print("Done.")

''' KNNBaseline hyperparameters tuning using RandomizedSearchCV() for both dataset1 and dataset2'''

#function for the tuning of the hyperparameters using the RandomizedSearchCV()
def tuning_KNNBaseline(data): 
    param_grid = { 
                "k":[5,10,20,25], #[20,40,50,60], 
                "min_k":[1,5,7,9,13],#[1,3,5,8,11,13,15,18,21],
                "sim_options":{
                               'name': ["cosine","pearson_baseline"],  
                               'user_based': [True, False],  
                               'min_support': [1,5,10,12]#[3,5,8,11,13]
                                },
                "bsl_options":{
                            'method': ['sgd'],
                            'learning_rate':[0.001,0.007,0.1],#[0.002,0.005,0.01],
                            'n_epochs':[20,30,40],#[50,100,150],
                            'reg': [0.01,0.03,0.06]#[0.01,0.02,0.05]
                            }
                }
    #to compute the execution time
    start=time.time()
    rcv = RandomizedSearchCV(KNNBaseline, param_grid, measures=['rmse'], cv=5,n_jobs=4) 
    rcv.fit(data)
    ex_time= round(time.time()-start,2)
    #store the best score obtained
    best_score = rcv.best_score['rmse']
    #store the best parameters
    best_param = rcv.best_params['rmse']
    return (ex_time,best_score,best_param)

#tuning the hyperpameters for the KNNBaseline() on dataset1
randcv_dt1 = tuning_KNNBaseline(data1)
time1 = randcv_dt1[0]
print("Execution time for dataset1: ", round(time1,2),'s')
print()
best_score1 = randcv_dt1[1]
print("Best score for dataset1: ",best_score1) #avg RMSE from KNNBaseline obtained on dataset1
print()
best_param1 = randcv_dt1[2] #list of the best params
print(best_param1)

#tuning the hyperpameters for the KNNBaseline() on dataset2
randcv_dt2 = tuning_KNNBaseline(data2)
time2 = randcv_dt2[0]
print("Execution time for dataset2: ", round(time2,2), 's')
print()
best_score2 = randcv_dt2[1]
print("Best score for dataset2: ",best_score2) #avg RMSE from KNNBaseline obtained on dataset2
print()
best_param2 = randcv_dt2[2] #list of the best params
print(best_param2)

''' SVD hyperparameters tuning using GridSearchCV() for both dataset1 and dataset2'''

def tuning_SVD(data):    
    param_grid = {"n_factors":[80,100,120,150],#[25,50,100,150],
                  "lr_all": [0.005,0.008,0.1,0.5],#[0.005,0.01,0.5,1],
                  "init_mean":[0.01,0.02,0.06,0.08],#[0.10,0.30,0.50,0.70],
                  "reg_all":[0.06,0.1,0.4,0.5]#[0.01,0.05,0.07,0.1]
                 }
    #to compute the execution time
    start=time.time()
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5,n_jobs=4) 
    gs.fit(data)
    #execution time
    ex_time= round(time.time()-start,2)
    #store the best score obtained
    best_score = gs.best_score['rmse']
    #store the best params configuration
    best_param = gs.best_params['rmse']
    return (ex_time,best_score,best_param)

#tuning the hyperpameters for the SVD() on dataset1
grid_cv1 = tuning_SVD(data1)
time1 = grid_cv1[0]
print("Execution time for dataset1: ", round(time1,2),'s')
print()
best_score1 = grid_cv1[1]
print("Best score for dataset1: ",best_score1) #avg RMSE from SVD obtained on dataset1
print()
best_param1 = grid_cv1[2] #list of the best params
print(best_param1)


#tuning the hyperpameters for the SVD() on dataset2
grid_cv2= tuning_SVD(data2)
time2 = grid_cv2[0]
print("Execution time for dataset2: ", round(time2,2),'s')
print()
best_score2 = grid_cv2[1]
print("Best score for dataset2: ",best_score2) #avg RMSE from SVD obtained on dataset1
print()
best_param2 = grid_cv2[2] #list of the best params
print(best_param2)
