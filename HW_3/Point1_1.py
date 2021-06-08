from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import os
import jsonlines

''' PROCESSING THE FILE ''' 

# Processing the file fever-dev-kilt.jsonl
with jsonlines.open("fever-dev-kilt.jsonl") as reader,\
        jsonlines.open("prepro_fever-dev-kilt.jsonl", "w") as writer:
    new_out_line = [{"answer":""}]
    line = ""
    for obj in reader:
        line = obj
        new_out_line[0]["answer"] = obj['output'][0]["answer"]
        line['output'] = new_out_line
        writer.write(line)
        
# Processing the file fever-train-kilt.jsonl
with jsonlines.open("fever-train-kilt.jsonl") as reader,\
        jsonlines.open("prepro_fever-train-kilt.jsonl", "w") as writer:
    new_out_line = [{"answer":""}]
    line = ""
    for obj in reader:
        line = obj
        new_out_line[0]["answer"] = obj['output'][0]["answer"]
        line['output'] = new_out_line
        writer.write(line)
    
 ''' EMBEDDING CREATION ''' 
model = SentenceTransformer('paraphrase-distilroberta-base-v1') #loading the model

''' DEV SET '''
dev_embeddings = []
with jsonlines.open('prepro_fever-dev-kilt.jsonl') as reader:
    for obj in reader:
      claims_dev = obj['input'] #extract the input content for the embedding
      embeddings = model.encode(claims_dev) #Get the embeddings for str
      obj['claim_embedding'] = embeddings.tolist() #add a new item (the embeddings list) in the dict
      dev_embeddings.append(obj) #store the new dict into a list for saving the result into a new json file
      
# save it into a file
with jsonlines.open('emb_dev.jsonl', mode = 'w') as writer:
    writer.write(dev_embeddings)
    
''' TRAIN SET '''
train_embeddings = []
with jsonlines.open('prepro_fever-train-kilt.jsonl') as reader:
    for obj in reader:
      claims_train = obj['input'] #extract the input content for the embedding
      embeddings = model.encode(claims_train) #Get the embeddings for str
      obj['claim_embedding'] = embeddings.tolist() #add a new item (the embeddings list) in the dict
      train_embeddings.append(obj) #store the new dict into a list for saving the result into a new json file

# save it into a file
with jsonlines.open('emb_train.jsonl', mode = 'w') as writer:
    writer.write(train_embeddings)
    
''' TEST SET '''
test_embeddings = []
with jsonlines.open('fever-test_without_answers-kilt.jsonl') as reader:
    for obj in reader:
      claims_test= obj['input'] #extract the input content for the embedding
      embeddings = model.encode(claims_test) #Get the embeddings for str
      obj['claim_embedding'] = embeddings.tolist() #add a new item (the embeddings list) in the dict
      test_embeddings.append(obj) #store the new dict into a list for saving the result into a new json file
      
# save it into a file
with jsonlines.open('emb_test.jsonl', mode = 'w') as writer:
    writer.write(test_embeddings)
