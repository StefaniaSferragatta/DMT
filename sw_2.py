from whoosh import index
import csv
import time

current_time_msec = lambda: int(round(time.time() * 1000))

###
### Open the Index
###
directory_containing_the_index = './directory_index_SimpleAnalyzer'
ix = index.open_dir(directory_containing_the_index)

###
### Fill the Index
###	
print("TimeStamp: ", time.asctime(time.localtime(time.time())))
ts_start = current_time_msec()
writer = ix.writer() #create a writer object
#
ALL_DOCUMENTS_file_name = "./lyrics_dataset/first_10K_lyrics_from_MetroLyrics.csv"
in_file = open(ALL_DOCUMENTS_file_name, "r", encoding='latin1')
csv_reader = csv.reader(in_file, delimiter=',')
csv_reader.__next__()  # to skip the header: first line containing the name of each field.
num_added_records_so_far = 0
for record in csv_reader:
    id = record[0] #extract the id doc
    lyrics = record[5] #extract the content
    #
    writer.add_document(id=id, content=lyrics) # add this new doc into the inverted index according to the schema
    #
    num_added_records_so_far += 1 
    if (num_added_records_so_far % 1000 == 0):
        print(" num_added_records_so_far= " + str(num_added_records_so_far))
#
writer.commit() #to commit all the operations on the file
in_file.close()
#
ts_end = current_time_msec()
print("TimeStamp: ", time.asctime(time.localtime(time.time())))
total_time_msec = (ts_end - ts_start)
print("total_time= " + str(total_time_msec) + "msec")
