import pandas as pd
import numpy as np
import math
import csv
from whoosh import *
from statistics import mean

'''Function used to read the .tsv file'''
def readfile(filename):
    tsv_file = open(filename)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    return list(read_tsv)


'''Score based on position (from Whoosh documentation) for the function scoring.FunctionWeighting(pos_score_fn)'''
def pos_score_fn(searcher, fieldname, text, matcher):
    poses = matcher.value_as("positions")
    return 1.0 / (poses[0] + 1)


''' Functions for the evaluation metrics'''
#function used in P@k to count the number of relevant doc at level k 
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
  
#nDCG
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
        pos = 1
        dic_rel = {}
        for it in seID[:k]:
            rel=1 if it in gtID else 0
            dic_rel[str(pos)]= rel
            pos+=1
        k1=k if k<len(seID[:k]) else len(seID[:k])
        for p in range(1,k1+1):
            dcg+=dic_rel[str(p)]/(math.log2(p+1))
            idcg+=1/(math.log2(j+1))
        ndcg.append(dcg/idcg)
    return (mean(ndcg))
