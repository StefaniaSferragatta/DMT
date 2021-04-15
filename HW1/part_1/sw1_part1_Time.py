import os
import csv
import time
import random
import numpy as np
import modin.pandas as pd
from bs4 import BeautifulSoup
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import *
from whoosh.qparser import *
from whoosh.writing import AsyncWriter
from whoosh import scoring
from whoosh import *
from Utilities_time import * #my script .py with the implementation of the evaluation metrics and utilities functions

'''Function to convert the html file into csv'''
def converter():
    #initialization of the dataframe for the time csv
    time_df=pd.DataFrame(columns=['ID','body']) 
    #for each doc in the folder Time_DATASET
    for j in range(1,423): 
        #define the file name
        filename2='_'*6+str(j)+'.html' 
        #open the file in reading mode
        with open(filename2) as f:
            content = f.read() 
        #use a html parser to pick the content of the file 
        soup = BeautifulSoup(content, 'html.parser')
        # save the body
        body=soup.body.string 
        #store into the dataframe the content. Each row of the document df is a .html file
        time_df=time_df.append({'ID':j,'body':body},ignore_index=True) 
    return time_df

# save the doc from Time dataset into csv file format
doc_Time_converted = converter()
doc_Time_converted.to_csv(os.getcwd()+"\Time_to_index.csv")

'''First part of the software for the search engines'''
def sw_1(analyzer,filename):
    # creating schema with fields id and content for the body
    schema = Schema(id=ID(stored=True),content=TEXT(stored=False, analyzer=analyzer))
    directory_containing_the_index = './directory_index'
    # create an empty-index according to the just defined schema in the directory where csv file is
    ix = create_in(directory_containing_the_index, schema) 
    # open the index file
    ix = index.open_dir(directory_containing_the_index) 
    #define a writer object to add content to the fields
    writer =  AsyncWriter(ix) 
    # fill the index:
    ALL_DOCUMENTS_file_name = filename #path of the file 
    # open the file and read it as csv
    in_file = open(ALL_DOCUMENTS_file_name, "r", encoding='latin1')
    csv_reader = csv.reader(in_file, delimiter=',')  
    # to skip the header: first line contains the name of each field.
    csv_reader.__next__()
    # for each row in the 'doc_to_index' file 
    for record in csv_reader: 
        id = record[1] # extract the id doc
        content = record[2] # extract the body
        # add this new doc into the inverted index according to the schema
        writer.add_document(id=id, content=' '+content)
    # commit all the operations on the file
    writer.commit()
    # close the file
    in_file.close()
    
'''Defing the query engine (second part of the software for the search engines)'''
def sw_2(analyzer,score_fun,input_query,max_number_of_results):
    directory_containing_the_index = './directory_index'
    # thanks to the ix we can retreive doc of interest for the given SE configurations
    ix = index.open_dir(directory_containing_the_index) 
    # define a QueryParser for parsing the input_query
    qp = QueryParser("content", ix.schema)
    # apply it on the given query
    parsed_query = qp.parse(input_query) 
    # create a Searcher for the Index with the given scoring function 
    searcher = ix.searcher(weighting=score_fun) 
    # store results of the query and limiting max number of results
    results = searcher.search(parsed_query,limit=max_number_of_results) 
    # define a dataframe to store the results 
    result=pd.DataFrame() 
    row=pd.DataFrame()
    for hit in results:
        row=pd.DataFrame([str(hit.rank) , int(hit['id']), str(hit.score)]).T
        result=result.append(row)
    result.columns=["Rank" , "Doc_ID" , "Score"] 
    # the column 'score' contains the values of the scoring function, that we use for having in a quantitative way the relevance of a doc for a particulare query
    searcher.close()
    return result

'''Execute the engine with the given configuration'''
def executor(analyzer,score_fun):
    result1=pd.DataFrame() # dataframe with the results of the given SE for ALL the queries; 
    tmp1=pd.DataFrame() #temporary dataframe 
    
    # open the file with all the queries
    Queries_file1= readfile(r"time_Queries.tsv")
    Queries=pd.DataFrame(Queries_file1[1:],columns=['Query_ID','Query'])
    
    # open the file of the GT
    gt_csv = readfile(r"time_Ground_Truth.tsv")
    gt1 = pd.DataFrame(gt_csv[1:],columns=['Query_id','Relevant_Doc_id'])

    #define a list with the unique query ids
    Q1=list(gt1['Query_id'].unique()) 
    dq1={} 
    # for each query_id
    for i in Q1: 
        #key=Query_id, value=number of relevant documents related to that query_id
        dq1[i]=len(list(gt1[gt1['Query_id']==i]['Relevant_Doc_id']))

    file_toindex1=os.getcwd()+"\Time_to_index.csv"
    # invoke the function to create the schema and to store the index file based on the retrieved 'doc_to_index.csv' file  
    sw_1(analyzer,file_toindex1)
    # for each index in the query set
    for i in Q1:
        # store the number of relevant documents related to the specific input query
        max_number_of_results_1q=dq1[i] 
        if max_number_of_results_1q==0:
            max_number_of_results_1q=1
        # invoke the function that,given the input query and given the specific SE configuration,
        # returns the results of the search and store it into a tmp dataframe
        tmp1=sw_2(analyzer,score_fun,list(Queries[Queries['Query_ID']==i]['Query'])[0],max_number_of_results_1q)
        tmp1['Query_id']=i
        result1=result1.append(tmp1) #add it to the result dataframe 
    return result1

'''Definition of the core search engine'''
def search_engine():
    # open the GT
    gt_csv = readfile(r"time_Ground_Truth.tsv")
    gt1 = pd.DataFrame(gt_csv[1:],columns=['Query_id','Relevant_Doc_id'])

    list_mrr1=[] # to store the MRR values for each SE configuration 
    # define the scoring functions
    score_functions = [scoring.FunctionWeighting(pos_score_fn),scoring.PL2(),scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)]
    # define the text analyzers
    analyzers = [StemmingAnalyzer(),FancyAnalyzer(),LanguageAnalyzer('en')]
    # store the name into lists to add it into the output SE_file together with the score
    analyz=['StemmingAnalyzer()','FancyAnalyzer()','LanguageAnalyzer()']
    scor_fun=[' FunctionWeighting',' PL2',' BM25F']
    #initialize a counter
    i=1
    #invoke the executor() with the combinations analyzer&scoring function
    for x in range(len(analyzers)):
        for y in range(len(score_functions)):
            print('Executing config n:' + str(i))
            # execute queries with the chosen configuration
            sr_1=executor(analyzers[x],score_functions[y]) 
            #save results of the search engine
            sr_1.to_csv("Time_DATASET"+str(i)+".csv",index=False) 
            
            #open the file and compute the MRR
            file_sr = open(r"Time_DATASET"+str(i)+".csv")
            se_csv = list(csv.reader(file_sr, delimiter=","))
            sr = pd.DataFrame(se_csv[1:],columns=['Rank','Doc_ID','Score','Query_id'])
            
            list_mrr1.append((analyz[x]+scor_fun[y],MRR(sr,gt1))) 
            i+=1
    # save into a table with MRR evaluation for every search engine configuration 
    mrrs=pd.DataFrame(list_mrr1)
    mrrs.to_csv("mrr.csv", index=False) #store MRR table
    
# exec the search engine with the different configurations for the Time dataset
search_engine()

'''Function for the creation od the distribution table'''
def r_distribution(num_configuration): 
    # Open the GT
    gt_csv = readfile(r"time_Ground_Truth.tsv")
    gt = pd.DataFrame(gt_csv[1:],columns=['Query_id','Relevant_Doc_id'])
    #define an empty list that will contanin the r_precision for each query in the GT
    r_list = []
    #compute the r-precision on each SE 
    for i in range(1,num_configuration+1): 
        #read the SE results for the time dataset
        file_se = open(r"Time_DATASET"+ str(i)+".csv")
        se_csv = list(csv.reader(file_se, delimiter=","))
        sr = pd.DataFrame(se_csv[1:],columns=['Rank','Doc_ID','Score','Query_id'])
        
        r_list.append(r_precision(sr,gt))
    #store the result into a df
    R_precision = pd.DataFrame(r_list)
    R_precision.index = np.arange(1, len(R_precision)+1) #reset the index and start from 1

    # Do the r-precision distribution table for each configuration in the Cranfield dataset
    metric= [R_precision.mean(axis=1),R_precision.min(axis=1),R_precision.quantile(0.25,axis=1),R_precision.median(axis=1),R_precision.quantile(0.75,axis=1),R_precision.max(axis=1)]
    r_distr = pd.DataFrame(metric).T
    #set the cols name
    r_distr.columns=['Mean','Min','1°_quartile','Median','3°_quartile','Max']
    #add a column to indicate the SE configuration
    configs = ['conf_1','conf_2','conf_3','conf_4','conf_5','conf_6','conf_7','conf_8','conf_9']
    r_distr['SE_Config'] = configs
    r_distri = r_distr[['SE_Config','Mean','Min','1°_quartile','Median','3°_quartile','Max']]
    r_distri.index = np.arange(1, len(configs)+1) #reset the index and start from 1
    return r_distri

#invoke the function and save the df into a csv file
r_pr_distr = r_distribution(9) 
r_pr_distr.to_csv('R_precision_distribution.csv') 

'''Function to store the top 5 configuration according to the MRR'''
def top_five():
    #open the file with the mrr and save it into a df
    file_MRR = open(r"mrr.csv")
    MRR_csv = list(csv.reader(file_MRR, delimiter=","))
    mrr_time = pd.DataFrame(MRR_csv[1:])
    #add cols name
    mrr_time.columns = ['Config','MRR'] 
    #change type of the column into float
    mrr = mrr_time.astype({"MRR": float}) 
    #reset a index
    mrr.index = np.arange(1, len(mrr)+1) 
    #sort it in ascending order and pick the top 5 SE 
    top = mrr.sort_values(by = ['MRR'],ascending=False).head(5)
    #save the index of the top 5 SE configuration and return them
    top_five = top.index
    return top_five

top_conf = top_five()
top_five = list(top_conf)

'''P@k on the top 5 configurations'''
def p_topfive(top,k):
    p_at_k_list =[]
    gt_csv = readfile(r"time_Ground_Truth.tsv")
    gt = pd.DataFrame(gt_csv[1:],columns=['Query_id','Relevant_Doc_id'])
    #for each index of the top 5
    for i in top:
        #read the SE results for the time dataset
        file_se = open(r"Time_DATASET"+ str(i)+".csv")
        se_csv = list(csv.reader(file_se, delimiter=","))
        sr = pd.DataFrame(se_csv[1:],columns=['Rank','Doc_ID','Score','Query_id'])
        #compute the P@k and store the result into a list
        p_at_k_list.append(p_at_k(sr,gt,k))
    return p_at_k_list

k_list = [1, 3, 5, 10]
output=[]
#invoke the function for each value of k
for k in k_list:
    output.append(p_topfive(top_five,k))
#save the result into a df
p_at_k_df = pd.DataFrame(output)
p_at_k_df.index = k_list #set the index
p_at_k_df.columns = ['SE_9','SE_3','SE_6','SE_8','SE_2'] #set the cols name with the top 5 SE configuration

#plot the P@k top5
plot1 = p_at_k_df.plot(y=['SE_9','SE_3','SE_6','SE_8','SE_2'],colormap="cool",\
              xlabel="k", ylabel="values",figsize=(10,10), title = 'P@k Time dataset').get_figure();
plot1.savefig('Time_p_plot.jpg')

'''nDCG on the top 5 configurations'''
def ndcg_topfive(top,k):
    ndcg_list =[]
    gt_csv = readfile(r"time_Ground_Truth.tsv")
    gt = pd.DataFrame(gt_csv[1:],columns=['Query_id','Relevant_Doc_id'])
    
    for i in top:
        #read the SE results for the time dataset
        file_se = open(r"Time_DATASET"+ str(i)+".csv")
        se_csv = list(csv.reader(file_se, delimiter=","))
        sr = pd.DataFrame(se_csv[1:],columns=['Rank','Doc_ID','Score','Query_id'])
        #compute the nDCG and store the result into a list
        ndcg_list.append(n_dcg(sr,gt,k))
    return ndcg_list

#invoke the function for each value of k
k_list = [1, 3, 5, 10]
output_ndcg=[]
for k in k_list:
    output_ndcg.append(ndcg_topfive(top_five,k))
#save the result into a df
ndcg_df = pd.DataFrame(output_ndcg)
ndcg_df.index = k_list #set the index
ndcg_df.columns = ['SE_9','SE_3','SE_6','SE_8','SE_2'] #set the cols name with the top 5 SE configuration

#plot the nDCG 
plot2 = ndcg_df.plot(y=['SE_9','SE_3','SE_6','SE_8','SE_2'],colormap="magma",\
            xlabel="k", ylabel="values",figsize=(10,10), title = 'nDCG@k Time dataset').get_figure();
plot2.savefig('Time_ndcg_plot.jpg')
