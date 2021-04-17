import pandas as pd
import re
import os

# -----------------------------------------------------------------------------
# ---------------------------------- my code ----------------------------------
# -----------------------------------------------------------------------------

# Creating & reading the dataset

path_ = os.getcwd()
data = pd.read_csv(path_ + "/dataset/250K_lyrics_from_MetroLyrics.csv", usecols=['ID', 'song'])
data.head()

# removing punctuation and converting everything into lowercase
data['ELEMENTS_IDS'] = pd.Series(re.sub(r'[^\w\s]', ' ', str(x).lower()) for x in data['song'])

# saving only IDs and lyrics as text columns
dataset = data[['ID', 'ELEMENTS_IDS']]


# changing the IDs to id_number


def get_ids(x):
	return 'id_' + str(x)
# for num in range(0, len(data['ID'])):
#     print('id_' + str(num))


# defining shingling

def get_my_shing(size, title):
	shing_list = set()

	if len(title) >= 3:
		for i in range(0, len(title)-size+1):
			shing_list.add(abs(hash(''.join(title[i:i+size])) % (10 ** 8)))
	else:
		shing_list.add(abs(hash(''.join(title[0:len(title)])) % (10 ** 8)))
	return shing_list


# getting shingles
dataset['ELEMENTS_IDS'] = [list(get_my_shing(3, dataset['ELEMENTS_IDS'][i].split())) for i in range(0, len(dataset['ELEMENTS_IDS']))]



# changing IDS to id_ID
dataset['ID'] = [(get_ids(dataset['ID'][i])) for i in range(0, len(dataset['ID']))]


# exporting dataset

dataset.to_csv(path_ + "/dataset/dataset_250k_title.tsv", sep='\t', index=False)
