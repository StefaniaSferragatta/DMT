import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eval_measures import *

'''Function to read tsv file'''
def readfile(filename):
    tsv_file = open(filename)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    return list(read_tsv)

# SE - from list to df
se1 = readfile("part_1_2__Results_SE_1.tsv")
se_1 = pd.DataFrame(se1[1:],columns=['Query_id','Doc_ID','Rank']) #convert the SE1 into a dataframe
se2 = readfile("part_1_2__Results_SE_2.tsv")
se_2 = pd.DataFrame(se2[1:],columns=['Query_id','Doc_ID','Rank']) #convert the SE2 into a dataframe
se3 = readfile("part_1_2__Results_SE_3.tsv")
se_3 = pd.DataFrame(se3[1:],columns=['Query_id','Doc_ID','Rank']) #convert the SE3 into a dataframe
# GT - from list to df
gt = readfile("part_1_2__Ground_Truth.tsv")
ground_truth = pd.DataFrame(gt[1:],columns=['Query_id','Relevant_Doc_id']) #convert the gt list into dataframe in order to extract the items in the col 'Query_id'

'''Function to apply the unranked eval measure on the SE. It returns a dataframe'''
def evaluation():
    #SE1
    p1 = precision(se_1, ground_truth)
    r1 = recall(se_1, ground_truth)
    f1 = f_measure(p1,r1)
    #SE2
    p2 = precision(se_2, ground_truth)
    r2 = recall(se_2, ground_truth)
    f2 = f_measure(p2,r2)
    #SE3
    p3 = precision(se_3, ground_truth)
    r3 = recall(se_3, ground_truth)
    f3 = f_measure(p3,r3)
    #store the results into a touple
    touple = [(p1,r1,f1),(p2,r2,f2),(p3,r3,f3)]
    
    #create a dataframe with the scores obtained
    df = pd.DataFrame(touple)
    df.columns=['%Precision','%Recall','%F1']
    #reset the index and start from 1
    df.index = np.arange(1, len(df)+1) #the indixes indicates the Search engines
    return df

#invoke the function to obtain the eval metrics for each SE
evaluation()

'''Function to apply the ranked eval measure (P@k) on the SE. It returns a dataframe'''
def evaluation_at_k():
    #invoke the P@k functione for each SE 
    pk1=round(mean(p_at_k(se_1, ground_truth,k=4))*100,2)
    pk2=round(mean(p_at_k(se_2, ground_truth,k=4))*100,2)
    pk3=round(mean(p_at_k(se_3, ground_truth,k=4))*100,2)
    touple_results = [(pk1,pk2,pk3)]
    #create a dataframe with the scores for each SE
    df_k = pd.DataFrame(touple_results).T
    df_k.columns=['%P@k']
    #reset the index and start from 1
    df_k.index = np.arange(1, len(df)+1) #the indixes indicates the Search engines
    return df_k

#invoke the function to obtain the P@k values for each SE
evaluation_at_k()
