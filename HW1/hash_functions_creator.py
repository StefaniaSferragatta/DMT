import pandas as pd
import re
import random
import math
import os

################################################
num_hash_functions = 10
upper_bound_on_number_of_distinct_elements  = 10000000
#upper_bound_on_number_of_distinct_elements =   138492
#upper_bound_on_number_of_distinct_elements =  3746518

################################################


### primality checker
def is_prime(number):
	if number == 2:
		return True
	if (number % 2) == 0:
		return False
	for j in range(3, int(math.sqrt(number)+1), 2):
		if (number % j) == 0:
			return False
	return True



set_of_all_hash_functions = set()
while len(set_of_all_hash_functions) < num_hash_functions:
	a = random.randint(1, upper_bound_on_number_of_distinct_elements-1)
	b = random.randint(0, upper_bound_on_number_of_distinct_elements-1)
	p = random.randint(upper_bound_on_number_of_distinct_elements, 10*upper_bound_on_number_of_distinct_elements)
	while is_prime(p) == False:
		p = random.randint(upper_bound_on_number_of_distinct_elements, 10*upper_bound_on_number_of_distinct_elements)
	#
	current_hash_function_id = tuple([a, b, p])
	set_of_all_hash_functions.add(current_hash_function_id)

print("a\tb\tp\tn")
for a, b, p in set_of_all_hash_functions:
	print(str(a) + "\t" + str(b) + "\t" + str(p) + "\t" + str(upper_bound_on_number_of_distinct_elements))

# -----------------------------------------------------------------------------
# ---------------------------------- my code ----------------------------------
# -----------------------------------------------------------------------------

# Creating & reading the dataset

path_ = os.getcwd()
dataset = pd.read_csv(path_ + "/dataset/250K_lyrics_from_MetroLyrics.csv", usecols=['ID', 'lyrics'])
dataset.head()

# removing punctuation and converting everything into lowercase
dataset['text'] = pd.Series(re.sub(r'[^\w\s]', '', x.lower()) for x in dataset['lyrics'])

# saving only IDs and lyrics as text columns
dataset = dataset[['ID', 'text']]

# defining shingling

def get_my_shing(size, lyric):
	shing_list = set()

	for i in range(0, len(lyric)-size+1):
		shing_list.add(abs(hash(''.join(lyric[i:i+size])) % (10 ** 8)))
	return shing_list


dataset['text'] = [list(get_my_shing(3, dataset['text'][i].split())) for i in range(0, len(dataset['text']))]

# exporting dataset

dataset.to_csv(path_ + "/dataset/dataset_250.tsv", sep='\t', index=False)


















