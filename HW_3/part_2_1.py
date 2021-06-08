from sentence_transformers import SentenceTransformer
from genre.hf_model import GENRE
import numpy as np
import pandas as pd
import pprint as pp
import jsonlines as jl
import json
import re
from datetime import datetime
import pytz
import torch


'''
things to install
pip install -U sentence-transformers
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/facebookresearch/GENRE.git
pip install transformers>=4.2.0
pip install jsonlines
pip install numpy
pip install pandas
pip install BeautifulSoup4

This code was executed on a PC with 16GB of RAM and 11GB of GPU,
and it took around 16 hours total.
Without the utilization of the GPU, meaning using only the CPU,
the code would have been running for at least 3 days!
'''

# checking if there is an available GPU to use with cuda
# "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" -l 5
# checking if there is cuda device available
print(torch.cuda.is_available())
# releasing the cache of the cuda device
torch.cuda.empty_cache()

# loading the GENRE model in the GPU
device = torch.device("cuda")
model = GENRE.from_pretrained("models/hf_e2e_entity_linking_wiki_abs").eval()
model.to(device)

# loading the model for sentence_transformer in the GPU
model_sent = SentenceTransformer('paraphrase-distilroberta-base-v1')
model_sent.to(device)

# the storage for both models is around 3,5 GB

# opening the wikipedia kilt knowledge
f = open("models/abstract_kilt_knowledgesource.json")
wiki = json.load(f)

# function of current time
def curr_time(string):
    tz_Rome = pytz.timezone('Europe/Rome')
    datetime_Rome = datetime.now(tz_Rome)
    return string + " time: " + datetime_Rome.strftime("%H:%M:%S")

# test dataset
# took around 1h:19m
print(curr_time("Started"))

with jl.open("embedded/emb_test.jsonl") as reader,\
        jl.open("embedded/emb_test2.jsonl", "w") as writer:

    for obj in reader:
        # POINT 1
        # taking only the input field for GENRE analysis
        # and adding a space at the beginning and removing the
        # full stop at the end of the sentence
        # and putting the sentence into a list
        sentence = [" " + obj['input'].strip(".")]

        # using the pretrained model (transformers)
        results = model.sample(sentence)

        # taking only the first result
        result = str(results[0][0]['text'])

        # listing all the wiki pages in a variable and adding that to the json row
        wiki_pages = re.findall(r'\[ *(.*?) *\]', result)

        obj['wikipedia_pages'] = wiki_pages

        # POINT 2

        # creating the empty list of the abstracts
        wiki_abst = []
        for arg in wiki_pages:
            # checking if the abstract exists
            if arg in wiki.keys():
                wiki_abst.append(wiki[arg].strip())
            # else putting a blank space in it
            else:
                wiki_abst.append(" ")

        # sorting in asc order the abstracts
        wiki_abst_sort = sorted(wiki_abst, key=len)

        # concatenating the abstracts separated by a whitespace
        wiki_abstract = ' '.join([str(item) for item in wiki_abst_sort])

        # adding the abstract string to the json row
        obj['wikipedia_abstract'] = wiki_abstract

        # POINT 3
        # setting the max_seq_length
        model_sent.max_seq_length = 256

        # getting the embeddings for the wikipedia_abstract
        embeddings = model_sent.encode(wiki_abstract)

        # adding the embeddings to the json row
        obj['abstract_embedding'] = embeddings.tolist()

        # POINT 4
        # storing the json obj in the new jsonl file
        writer.write(obj)

print(curr_time("Finished"))

# dev dataset
# took around 1h:20m
print(curr_time("Started"))

with jl.open("embedded/emb_dev.jsonl") as reader,\
        jl.open("embedded/emb_dev2.jsonl", "w") as writer:

    for obj in reader:

        # POINT 1
        # taking only the input field for GENRE analysis
        # and adding a space at the beginning and removing the
        # full stop at the end of the sentence
        # and putting the sentence into a list
        sentence = [" " + obj['input'].strip(".")]

        # using the pretrained model (transformers)
        results = model.sample(sentence)

        # taking only the first result
        result = str(results[0][0]['text'])

        # listing all the wiki pages in a variable and adding that to the json row
        wiki_pages = re.findall(r'\[ *(.*?) *\]', result)

        obj['wikipedia_pages'] = wiki_pages

        # POINT 2

        # creating the empty list of the abstracts
        wiki_abst = []
        for arg in wiki_pages:
            # checking if the abstract exists
            if arg in wiki.keys():
                wiki_abst.append(wiki[arg].strip())
            # else putting a blank space in it
            else:
                wiki_abst.append(" ")

        # sorting in asc order the abstracts
        wiki_abst_sort = sorted(wiki_abst, key=len)

        # concatenating the abstracts separated by a whitespace
        wiki_abstract = ' '.join([str(item) for item in wiki_abst_sort])

        # adding the abstract string to the json row
        obj['wikipedia_abstract'] = wiki_abstract

        # POINT 3
        # setting the max_seq_length
        model_sent.max_seq_length = 256

        # getting the embeddings for the wikipedia_abstract
        embeddings = model_sent.encode(wiki_abstract)

        # adding the embeddings to the json row
        obj['abstract_embedding'] = embeddings.tolist()

        # POINT 4
        # storing the json obj in the new jsonl file
        writer.write(obj)


print(curr_time("Finished"))

# train dataset
# took around 13h:32m
print(curr_time("Started"))

with jl.open("embedded/emb_train.jsonl") as reader,\
        jl.open("embedded/emb_train2.jsonl", "w") as writer:

    for obj in reader:

        # POINT 1
        # taking only the input field for GENRE analysis
        # and adding a space at the beginning and removing the
        # full stop at the end of the sentence
        # and putting the sentence into a list
        sentence = [" " + obj['input'].strip(".")]

        # using the pretrained model (transformers)
        results = model.sample(sentence)

        # taking only the first result
        result = str(results[0][0]['text'])

        # listing all the wiki pages in a variable and adding that to the json row
        wiki_pages = re.findall(r'\[ *(.*?) *\]', result)

        obj['wikipedia_pages'] = wiki_pages

        # POINT 2

        # creating the empty list of the abstracts
        wiki_abst = []
        for arg in wiki_pages:
            # checking if the abstract exists
            if arg in wiki.keys():
                wiki_abst.append(wiki[arg].strip())
            # else putting a blank space in it
            else:
                wiki_abst.append(" ")

        # sorting in asc order the abstracts
        wiki_abst_sort = sorted(wiki_abst, key=len)

        # concatenating the abstracts separated by a whitespace
        wiki_abstract = ' '.join([str(item) for item in wiki_abst_sort])

        # adding the abstract string to the json row
        obj['wikipedia_abstract'] = wiki_abstract

        # POINT 3
        # setting the max_seq_length
        model_sent.max_seq_length = 256

        # getting the embeddings for the wikipedia_abstract
        embeddings = model_sent.encode(wiki_abstract)

        # adding the embeddings to the json row
        obj['abstract_embedding'] = embeddings.tolist()

        # POINT 4
        # storing the json obj in the new jsonl file
        writer.write(obj)


print(curr_time("Finished"))

f.close()
