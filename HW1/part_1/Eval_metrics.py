import pandas as pd
import numpy as np
import math

# P@k 
def p_at_k(sr1,gt,k):
    #n_query = sr1['Query_id'].max()
    Q=set(sr1['Query_id'])
    for i in Q:#range(1, n_query+1):
        # filter the rows of the df with QUERY_ID = i
        seID = sr1['Doc_ID'].loc[sr1['Query_id'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        #because not all queries are considered in the Ground Truth CSV
        if len(gtID) == 0:
            continue
        numerator = sum(el in seID for el in gtID) #do the summation of the relevant doc in the SE
        denominator = min(k,len(gtID))
    return (numerator/denominator)

# R-precision
def r_precision(se, gt):
    r_pre=[]
    #n_query = se['Query_id'].max()
    Q=set(gt['Query_id'].unique())
    for i in Q:#range (1, n_query+1):
        seID = se['Doc_ID'].loc[se['Query_id'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        # because not all queries are considered in the Ground Truth CSV
        if len(gtID) == 0:
            continue
        num = sum(el in seID for el in gtID) #do the summation of the relevant doc in the SEengine
        r = num/(len(gtID))
        r_pre.append(r)
    return (r_pre)

# Mean Reciprocal Rank (MRR)
def MRR(sr1,gt):
    mrr=0
    rel_doc_ids=list() #list to store the relevant doc_ids for every query id in Q
    dd_se= {} 
    dd_gt= {} 
    Q=set(gt['Query_id'].unique()) #number of unique queries in the ground truth
    for i in Q:
        #key=Query_id,value=list of document ids from SE result
        dd_se[i]=list(sr1[sr1['Query_id']==i]['Doc_ID'])
    for i in Q:
        #key=Query_id,value=list of relevant document ids from ground truth
        dd_gt[i]=list(gt[gt['Query_id']==i]['Relevant_Doc_id'])
    
    for q in Q: 
        rel_doc_ids=dd_gt[q]
        #for each doc_id in query_id q
        for i in range(len(dd_se[q])): 
            #if doc_id is in the list of the relevant doc_ids 
            if dd_se[q][i] in rel_doc_ids: #[i] is index of list
                mrr=mrr+(1/(i+1)) #MRR value is sum on Reciprocal Ranks (+1 cause ranking starts with 1)
                break #once we get the first doc id from the relevant doc ids in the GT, we can stop
    mrr=mrr/(len(Q)) #compute the avg of the sum of reciprocal ranks
    return mrr

# normalized Discounted Cumulative Gain (nDCG)
def n_dcg(se,gt,k):
    result = 0
    #n_query = se['Query_id'].max()
    #for i in range(1,n_query+1):
    Q=set(gt['Query_id'].unique()) #number of unique queries in the ground truth
    for i in Q:
        top_result = pd.DataFrame()
        # get top k relevant docs from the SE1
        top_se = se[['Doc_ID', 'Rank']][:k]
        top_gt = gt[['Relevant_Doc_id']][:k]
        # store into top_result only relevant docs
        top_result = top_gt.merge(top_se, how='inner', left_on='Relevant_Doc_id', right_on='Doc_ID') 
        # (every element in the list has relevance = 1)
        rank = top_result['Rank'].tolist() # store into a list the ranks
        ranks = [int(i) for i in rank] #convert into int
        # compute the discounted cumulative gain
        dcg = 0
        for i in ranks: #if the query id is in the GT than relevance = 1
            dcg += 1/(math.log2(i+2))
        # compute the ideal discounted cumulative gain
        idcg = 0
        for j in range(1, k+1):
            idcg += 1/(math.log2(j+2))
        return (dcg/idcg)