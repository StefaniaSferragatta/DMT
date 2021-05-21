from surprise import Dataset,Reader 
from surprise import NormalPredictor,BaselineOnly
from surprise import KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline
from surprise import SVD,SVDpp,NMF,SlopeOne
from surprise import CoClustering
from surprise.model_selection import KFold, cross_validate
import pandas as pd
import numpy as np
import time
import os

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

def Recommendation_System(data):
    # list of the algorithms and their name
    algorithms= [NormalPredictor,BaselineOnly,KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline,SVD,SVDpp,NMF,SlopeOne,CoClustering]
    algo_names= ['NormalPredictor','BaselineOnly','KNNBasic','KNNWithMeans','KNNWithZScore','KNNBaseline','SVD','SVDpp','NMF','SlopeOne','CoClustering]
    #useful data stracture
    row=[]
    table_df = pd.DataFrame()
    
    #set the folds for the CV
    kf = KFold(n_splits=5, random_state=0)
    
    #for each algorithm in the list
    for current_algo in range(len(algorithms)):
        result=cross_validate(algorithms[current_algo](), data, measures=['RMSE'], cv=kf, verbose=True,n_jobs=4)
        print()
        #compute the mean
        mean_rmse = '{:.4f}'.format(np.mean(result['test_rmse']))  
        #add the result to a list
        row.append([algo_names[current_algo], mean_rmse])
        
    #fill the dataframe with the results obtained 
    table_df=table_df.append(row)
    #set the cols name
    table_df.columns =['Algorithm', 'RMSE']
    #sort the dataframe 
    table_df.sort_values(by=['RMSE'], inplace=True, ascending=True)
                 
    return table_df
    
#invoke the function for the first dataset
Recommendation_System(data1)
#invoke the function for the second dataset
Recommendation_System(data2)
