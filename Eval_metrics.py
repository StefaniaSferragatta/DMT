## Evaluationt metrics
import csv
import pandas as pd
import numpy as np
import pylab
import matplotlib

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

#Ground truth
gt = readfile("Ground_Truth.tsv")

# Precision at k (P@k)
length_gt = len(gt[1:])  # size of the Ground Truth
k = 18
#denominator
den = min(k,length_gt)
# P@k for SE_1
se1 = readfile("Results_from_SE1.tsv")
# numerator
num = relevant_docs(se1,gt,k)
P_at_k_SE1 = num/den
print('P@k for SE1: ', round(P_at_k_SE1,3))

#numerator SE_2
se2 = readfile("Results_from_SE2.tsv")
# P@k for SE_2
# numerator
num = relevant_docs(se2,gt,k)
P_at_k_SE2 = num/den
print('P@k for SE2: ', round(P_at_k_SE2,3)) 

#numerator SE_3
se3 = readfile("Results_from_SE3.tsv")
# numerator
num = relevant_docs(se3,gt,k)
P_at_k_SE3 = num/den
print('P@k for SE3: ', round(P_at_k_SE3,3))

#plot P@k for SE1 with different k 
k_list = (2,4,5,8,11,14,18,20,959)
dens = [] #denominators
nums = [] #numerators
for i in k_list:
    dens.append(min(i,length_gt))
    nums.append(relevant_docs(se1,gt,i))

p_at_k = [] #empty list for the results to plot
for n in nums:
    for d in dens:
        p_at_k.append(n/d)

pylab.figure(figsize=(10,10)) 
pylab.plot(p_at_k)
pylab.xlabel('K values')
pylab.ylabel('Scores')
pylab.title('P@k for SE1 with different k')
pylab.show()
#pylab.savefig('/P@k_se1.png')

# R-Precision
# SE1
num = relevant_docs(se1,gt,length_gt)
den = length_gt
r_precisonSE1 = num/den
print('R-precision for SE1: ', round(r_precisonSE1,3))
#SE2
num = relevant_docs(se2,gt,length_gt)
den = length_gt
r_precisonSE2 = num/den
print('R-precision for SE2: ', round(r_precisonSE2,3))
#SE3
num = relevant_docs(se3,gt,length_gt)
den = length_gt
r_precisonSE3 = num/den
print('R-precision for SE3: ', round(r_precisonSE3,3))

# Mean Reciprocal Rank (MRR)
# SE - list to df
se_1 = pd.DataFrame(se1,columns=['Query_id','Doc_ID','Rank','Score']) #convert the SE1 into a dataframe
se_2 = pd.DataFrame(se2,columns=['Query_id','Doc_ID','Rank','Score']) #convert the SE2 into a dataframe
se_3 = pd.DataFrame(se3,columns=['Query_id','Doc_ID','Rank','Score']) #convert the SE3 into a dataframe
# GT - list to df
ground_truth = pd.DataFrame(gt,columns=['Query_id','Relevant_Doc_id'])#convert the gt list into dataframe in order to extract the items in the col 'Query_id'
Q = set(ground_truth['Query_id'].unique()) #unique queries in the GT

#function to evaluate the MRR 
def MRR(Q,se,g_t):
    mrr = 0
    dd_se = {} #key=query_id; value= list of doc_id from the SE
    for i in Q:
        dd_se[i]=list(se[se['Query_id']==i]['Doc_ID'])

    dd_gt={} #key=Query_id; value=list of relevant document ids from ground truth
    for i in Q:
        dd_gt[i]=list(g_t[g_t['Query_id']==i]['Relevant_Doc_id'])

    rel_doc_ids=list() #list to store the relevant doc_ids for every query id in Q
    for q in Q: 
        rel_doc_ids=dd_gt[q]
        #for each doc_id in query_id q
        for i in range(len(dd_se[q])): 
            #if doc_id is in the list of the relevant doc_ids 
            if dd_se[q][i] in rel_doc_ids: #[i] is index of list
                mrr=mrr+(1/(i+1))	#mrr value is sum on Reciprocal Ranks (+1 cause ranking starts with 1)
                break #if it is break cause it found the first doc id from the relevant doc ids in the ground truth
    mrr=mrr/(len(Q)) #mean of the sum of reciprocal ranks
    return mrr

mrr1 = MRR(Q,se_1,ground_truth)
print('MRR for SE1: ', round(mrr1,3))
mrr2 = MRR(Q,se_2,ground_truth)
print('MRR for SE2: ', round(mrr2,3))
mrr3 = MRR(Q,se_3,ground_truth)
print('MRR for SE3: ', round(mrr3,3))

# normalized Discounted Cumulative Gain (nDCG)
'''
dq= {} #key=query_id, value=list of Relevant_Doc_ids
for i in Q:
    dq[i]=list(ground_truth[ground_truth['Query_id']==i]['Relevant_Doc_id']) 

dd= {}
for i in Q:
    dd[i]=list(se_1[se_1['Query_id']==i]['Doc_ID'])
ser1 = dd

# nDCG(q,k)=DCG(q,k)/IDCG(q,k)
def DCG(query_r,query_gt,k):
	s=0
	dsg=[]
	for p in range(len(query_r[0:k+1])):
        #compute the relevance of the doc 
		if query_r[p] in query_gt[0:k+1]:
			s=1 #1 if the doc_id belongs to the gt(query)
		else:
			s=0 #0 otherwise
		dsg.append(s/np.log2(p+1)) #+1 to avoid 0 at denominator
	return sum(dsg)


def myDCG_for_ser(ser_list,all_dict,dq,k):
	s_d=pd.DataFrame()
	for i in range(len(ser_list)):
		dcg=[]
		w_dict=all_dict[i]
		for y in w_dict.keys():
			dcg.append(DCG(w_dict[y],dq[y],k))
		s_d_temp=pd.DataFrame(dcg).T
		s_d=s_d.append(s_d_temp)
	return s_d

def nDCG(disc_cum_gain):
	for i in range(disc_cum_gain.shape[1]):
		if (np.max(disc_cum_gain[i])>0):
			disc_cum_gain[i]=disc_cum_gain[i]/np.max(disc_cum_gain[i])
	return disc_cum_gain

#creating DataFrame with nDCG values
k_nDCG=pd.DataFrame()
for k in range(1,len(list_of_names)):
	disc_cum_gain=myDCG_for_ser(list_of_ser,all_dicts,dq,k)
	nDCG=n_myDCG(disc_cum_gain)
	avgs=avg_s_dcg(nDCG)
	k_nDCG[k]=avgs
k_nDCG=k_nDCG.T
print(k_nDCG)
'''