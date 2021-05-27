''' Point 2 of Part 1.1'''

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np 
import jsonlines

'''Use the sentence-transformers python library to get sentence embeddings for each claim.'''

model = SentenceTransformer('paraphrase-distilroberta-base-v1') #loading the model

''' Embedding the DEV SET'''
dev_embeddings = []
with jsonlines.open('processed-dev-kilt.jsonl') as reader:
    for obj in reader:
      claims_dev=str(obj) #convert the list of dict into list of str for the embeddings
      embeddings = model.encode(claims_dev) #Get the embeddings for str
      for claim in obj: #for each dict in the .jsonl file
        claim['claim_embedding'] = embeddings.tolist() #add a new item (the embeddings list) in the dict
        dev_embeddings.append(claim) #store the new dict into a list for saving the result into a new json file
      
# save it into a file as requested
with jsonlines.open('emb_dev.jsonl', mode = 'w') as writer:
    writer.write(dev_embeddings)
    
    
''' Embedding the TRAIN SET'''
train_embeddings = []
with jsonlines.open('processed-train-kilt.jsonl') as reader:
    for obj in reader:
      claims_train=str(obj) #convert the list of dict into list of str for the embeddings
      embeddings = model.encode(claims_train) #Get the embeddings for str
      for claim in obj: #for each dict in the .jsonl file
        claim['claim_embedding'] = embeddings.tolist() #add a new item (the embeddings list) in the dict
        train_embeddings.append(claim) #store the new dict into a list for saving the result into a new json file
        
# save it into a file as requested
with jsonlines.open('emb_train.jsonl', mode = 'w') as writer:
    writer.write(train_embeddings)
    
    
''' Embedding the TEST SET'''
test_embeddings = []
with jsonlines.open('fever-test_without_answers-kilt.jsonl') as reader:
    for obj in reader:
      claims_test=str(obj) #convert the list of dict into list of str for the embeddings
      embeddings = model.encode(claims_test) #Get the embeddings for str
      obj['claim_embedding'] = embeddings.tolist() #add a new item (the embeddings list) in the dict
      test_embeddings.append(obj) #store the new dict into a list for saving the result into a new json file
    
# save it into a file as requested
with jsonlines.open('emb_test.jsonl', mode = 'w') as writer:
    writer.write(test_embeddings)
