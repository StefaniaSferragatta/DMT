from whoosh import index
from whoosh.qparser import *
from whoosh import scoring

#QUERY ENGINE
print()

###
### Open the Index
###
directory_containing_the_index = './directory_index_SimpleAnalyzer'
ix = index.open_dir(directory_containing_the_index) #thanks to it we can retreive doc of interest

###
### Create a Query
###
input_query = 'hate OR love'
max_number_of_results = 5 #top 5 (according to the relevance to the query itself)

###
### Select a Scoring-Function
###
scoring_function = scoring.Frequency() #tf-idf of the term in the doc

###
### Create a QueryParser for 
### parsing the input_query.
###
qp = QueryParser("content", ix.schema)
parsed_query = qp.parse(input_query)  # parsing the query
print("Input Query : " + input_query)
print("Parsed Query: " + str(parsed_query))

###
### Create a Searcher for the Index
### with the selected Scoring-Function 
###
searcher = ix.searcher(weighting=scoring_function)

### perform a Search :)
results = searcher.search(parsed_query, limit=max_number_of_results)

### print the ID of the best documents 
### associated to the input query.
print()
print("Rank" + "\t" + "DocID" + "\t" + "Score")
for hit in results[0]:
    print(str(hit.rank + 1) + "\t" + hit['id'] + "\t" + str(hit.score))
searcher.close()
#the score is the value of the scoring function that we use for having in a quantitative way the relevance of a doc for a particulare query
print()