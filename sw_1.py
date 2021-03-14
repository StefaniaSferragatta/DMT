from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import SimpleAnalyzer
# TO RUN: Lab_1 % time python3 -u sw_1.py
###
### Define a Text-Analyzer 
###
selected_analyzer = SimpleAnalyzer()

###
### Create a Schema 
###
# The schema it's a collection of fields, like a dictionary (key - values)
# ID = document identifier (id of the song). The 'store' flag forces to insert into the interted index only the doc id
# content = process the field according to the SimpleAnalyzer (lyric of the song)
schema = Schema(id=ID(stored=True), \
                content=TEXT(stored=False, analyzer=selected_analyzer))
###
### Create an empty-Index 
### according to the just defined Schema 
### 
directory_containing_the_index = './directory_index_SimpleAnalyzer'
create_in(directory_containing_the_index, schema)
