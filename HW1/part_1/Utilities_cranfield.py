import modin.pandas as pd
import numpy as np
import math
from whoosh import *
from statistics import mean

# Score based on position (from Whoosh documentation)
def pos_score_fn(searcher, fieldname, text, matcher):
    poses = matcher.value_as("positions")
    return 1.0 / (poses[0] + 1)

#function used in P@k and r_precision to count the number of relevant doc at level k 
def relevant_docs(search_engine,ground_truth,k):
    rel_doc = 0
    for i in search_engine[:k]:
        for j in ground_truth:
            if i==j:
                rel_doc +=1
    return rel_doc

# P@k 
def p_at_k(se,gt,k):
    p_list = []
    Q=set(gt['Query_id'].unique())
    for i in Q:
        seID = se['Doc_ID'].loc[se['Query_id'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        num = relevant_docs(seID,gtID,k) # numerator
        den = min(k,len(gtID)) # denominator
        p_list.append(num/den)
    return (mean(p_list))

# R-precision
def r_precision(se, gt):
    r_pre=[]
    Q=set(gt['Query_id'].unique())
    for i in Q:
        seID = se['Doc_ID'].loc[se['Query_id'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        k= len(gtID)
        num = relevant_docs(seID,gtID,k)
        r = num/(k)
        r_pre.append(r)
    return (r_pre)

# Mean Reciprocal Rank (MRR)
def MRR(sr1,gt):
    mrr=0
    relevant_doc=[] #list to store the relevant doc_ids for every query id in Q
    dd_se= {} 
    dd_gt= {} 
    Q=set(gt['Query_id'].unique()) #number of unique queries in the ground truth
    for i in Q:
        #key=Query_id,value=list of document ids from SE result
        dd_se[i]=sr1[sr1['Query_id']==i]['Doc_ID'].tolist()
        #key=Query_id,value=list of relevant document ids from ground truth
        dd_gt[i]=gt[gt['Query_id']==i]['Relevant_Doc_id'].tolist()
    
    for q in Q: 
        relevant_doc=dd_gt[q]
        #for each doc in the set of queries
        for i in range(len(dd_se[q])): 
            #if the doc_id is in the list of the relevant doc
            if dd_se[q][i] in relevant_doc: 
                mrr+=(1/(i+1)) #update the MRR value (+1 cause ranking starts with 1)
                break #once we get the first doc id from the relevant doc ids in the GT, we can stop
    mrr=mrr/(len(Q)) #compute the avg of the sum of reciprocal ranks
    return mrr

# normalized Discounted Cumulative Gain (nDCG)
def n_dcg(se,gt,k):
    rel=0
    dcg=0
    idcg=0
    ndcg=[]
    Q=set(gt['Query_id'].unique())
    #for each query in the GT
    for i in Q:
        #create two lists one for the se one for the gt relative to that query id
        seID = se['Doc_ID'].loc[se['Query_id'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        for el in gtID:
            for it in seID[:k]:
                #if the doc_id is in the gt then rel=1, else rel=0
                rel=1 if it==el else 0
        for p in range(1,k+1):
            dcg+=rel/(math.log2(p+1))
        for j in seID[:k]:
            idcg+=1/(math.log2(j+1))
        ndcg.append(dcg/idcg)
    return (mean(ndcg))