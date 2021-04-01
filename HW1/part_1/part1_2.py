import csv
import pandas as pd
import numpy as np
import pylab
import matplotlib
import math

# Usefull function ---------------------------------
# function to read tsv file
def readfile(filename):
    tsv_file = open(filename)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    return list(read_tsv)

#function to count the relevant docs
def relevant_docs(search_engine,ground_truth,k):
    rel_doc = 0
    for i in search_engine[1:k+1]:
        for j in ground_truth[1:]:
            if i[1]==j[1]:
                rel_doc +=1
    return rel_doc

# ----------- DEFINITION OF THE DATAFRAMES -----------------
# SE - from list to df
se1 = readfile("part_1_2__Results_SE_1.tsv")
se_1 = pd.DataFrame(se1[1:],columns=['Query_ID','Doc_ID','Rank']) #convert the SE1 into a dataframe
se2 = readfile("part_1_2__Results_SE_2.tsv")
se_2 = pd.DataFrame(se2[1:],columns=['Query_ID','Doc_ID','Rank']) #convert the SE2 into a dataframe
se3 = readfile("part_1_2__Results_SE_3.tsv")
se_3 = pd.DataFrame(se3[1:],columns=['Query_ID','Doc_ID','Rank']) #convert the SE3 into a dataframe
# GT - from list to df
gt = readfile("part_1_2__Ground_Truth.tsv")
ground_truth = pd.DataFrame(gt[1:],columns=['Query_ID','Relevant_Doc_id']) #convert the gt list into dataframe in order to extract the items in the col 'Query_id'
# -----------------------------------------------------------

## EVALUATION METRICS
# Precision at k (P@k) SE_1
length_gt = len(gt[1:])  # size of the Ground Truth
k = 18 #arbitrary
den = min(k,length_gt) # denominator
num = relevant_docs(se1,gt,k) # numerator
P_at_k_SE1 = num/den
print('P@k for SE1: ', P_at_k_SE1)


# P@k for SE_2
num = relevant_docs(se2,gt,k)
P_at_k_SE2 = num/den
print('P@k for SE2: ',P_at_k_SE2) 

# P@k for SE_3
num = relevant_docs(se3,gt,k)
P_at_k_SE3 = num/den
print('P@k for SE3: ', P_at_k_SE3)

#----------------------------------------
#Plot P@k for SE with different k 
# def plot_P_at_k(se):
#     k_list =sorted((np.random.randint(2, 1000, size=10)).tolist())
#     dens = [] #denominators
#     nums = [] #numerators
#     for i in k_list:
#         dens.append(min(i,length_gt))
#         nums.append(relevant_docs(se,gt,i))

#     p_at_k = [] #empty list for the results to plot
#     for n in nums:
#         for d in dens:
#             p_at_k.append(n/d)

#     pylab.figure(figsize=(10,10)) 
#     pylab.plot(p_at_k)
#     pylab.xlabel('K values')
#     pylab.ylabel('Scores')
#     pylab.title('P@k for SE with different k')
#     pylab.show()
#     #pylab.savefig('/P@k_se1.png')

# plot_P_at_k(se1)
# --------------------------------------

# R-Precision
# SE1
num = relevant_docs(se1,gt,length_gt)
den = length_gt
r_precisonSE1 = num/den
print('R-precision for SE1: ', r_precisonSE1)


#SE2
num = relevant_docs(se2,gt,length_gt)
den = length_gt
r_precisonSE2 = num/den
print('R-precision for SE2: ', r_precisonSE2)
#SE3
num = relevant_docs(se3,gt,length_gt)
den = length_gt
r_precisonSE3 = num/den
print('R-precision for SE3: ', r_precisonSE3)


# Mean Reciprocal Rank (MRR)
Q = set(ground_truth['Query_ID'].unique()) #unique queries in the GT

#function to evaluate the MRR (it works on dataframe - because it seems easer for me to pick the index)
def MRR(Q,se,g_t):
    mrr = 0
    dd_se = {} #key=query_id; value= list of doc_id from the SE
    for i in Q:
        dd_se[i]=list(se[se['Query_ID']==i]['Doc_ID'])

    dd_gt={} #key=Query_id; value=list of relevant document ids from GT
    for i in Q:
        dd_gt[i]=list(g_t[g_t['Query_ID']==i]['Relevant_Doc_id'])

    rel_doc_ids=list() #list to store the relevant doc_ids for every query id in Q
    for q in Q: 
        rel_doc_ids=dd_gt[q]
        #for each doc_id in query_id q
        for i in range(len(dd_se[q])): 
            #if doc_id is in the list of the relevant doc_ids 
            if dd_se[q][i] in rel_doc_ids: #[i] is index of list
                mrr=mrr+(1/(i+1))	#MRR value is sum on Reciprocal Ranks (+1 cause ranking starts with 1)
                break #once we get the first doc id from the relevant doc ids in the GT, we can stop
    mrr=mrr/(len(Q)) #compute the avg of the sum of reciprocal ranks
    return mrr

mrr1 = MRR(Q,se_1,ground_truth)
print('MRR for SE1: ', mrr1)

mrr2 = MRR(Q,se_2,ground_truth)
print('MRR for SE2: ', mrr2)
mrr3 = MRR(Q,se_3,ground_truth)
print('MRR for SE3: ', mrr3)

# normalized Discounted Cumulative Gain (nDCG)
def n_dcg(se,gt,k):
    result = 0
    n_query = int(se['Query_ID'].max())
    for i in range(1,n_query+1):
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
            dcg += 1/(math.log2(i+1))
        # compute the ideal discounted cumulative gain
        idcg = 0
        for j in range(1, k+1):
            idcg += 1/(math.log2(j+1))
        return (round(dcg/idcg, 5))

k = 18
ndcg1 = n_dcg(se_1,ground_truth,k)
print('nDCG for SE1: ', ndcg1)
ndcg2 = n_dcg(se_2,ground_truth,k)
print('nDCG for SE2: ', ndcg2)
ndcg3 = n_dcg(se_2,ground_truth,k)
print('nDCG for SE3: ', ndcg3)


# TO ADD PLOT AND CHANGE K 
