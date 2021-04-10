import numpy as np
import math
from statistics import mean

# function to count the number of relevant items retrieved
def relevant_docs(search_engine,ground_truth):
    rel_doc = 0
    for i in search_engine:
        for j in ground_truth:
            if i==j:
                rel_doc +=1
    return rel_doc

# precision function 
def precision(se,gt):
    p_list = []
    Q=set(gt['Query_id'].unique())
    for i in Q:
        seID = se['Doc_ID'].loc[se['Query_id'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        num = relevant_docs(seID,gtID) 
        p_list.append(num/len(seID)) # p= (#relevant items retrieved)/(#retrieved items)
        p = round(mean(p_list)*100,2)
    return (p)

# recall function 
def recall(se, gt):
    r_pre=[]
    Q=set(gt['Query_id'].unique())
    for i in Q:
        seID = se['Doc_ID'].loc[se['Query_id'] == i].tolist()
        gtID = gt['Relevant_Doc_id'].loc[gt['Query_id'] == i].tolist()
        num = relevant_docs(seID,gtID)
        r_pre.append(num/len(gtID)) # r = (#relevant items retrieved)/(#relevant items)
        r = round(mean(r_pre)*100,2)
    return (r)

# F measure
def f_measure(p,r):
    f = round(2/((1/p)+(1/r)),2)
    return f

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