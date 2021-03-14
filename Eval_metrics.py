## Evaluationt metrics
import csv
import numpy as np

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
print('P@k for SE2: ', round(P_at_k_SE2,3)) # equal to SE1, is normal?

#numerator SE_3
se3 = readfile("Results_from_SE3.tsv")
# P@k for SE_2
se3 = readfile("Results_from_SE3.tsv")
# numerator
num = relevant_docs(se3,gt,k)
P_at_k_SE3 = num/den
print('P@k for SE3: ', round(P_at_k_SE3,3))

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
"""
rec_rank=[]
for i in gt[1:]:
    for j in se1[1:]:
        pos=j[int(i[0])+1]
        RR=1/pos
        rec_rank.append(RR)
MRR_SE1=np.mean(rec_rank)
print(MRR_SE1)
"""


# normalized Discounted Cumulative Gain (nDCG)
