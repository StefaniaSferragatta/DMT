import pandas as pd
import re
import os

# -----------------------------------------------------------------------------
# ---------------------------------- my code ----------------------------------
# -----------------------------------------------------------------------------

# Creating & reading the dataset
path_ = os.getcwd()
data = pd.read_csv(path_ + "/dataset/250K_lyrics_from_MetroLyrics.csv", usecols=['ID', 'lyrics'])
data.head()


# removing punctuation and converting everything into lowercase
data['ELEMENTS_IDS'] = pd.Series(re.sub(r'[^\w\s]', '', x.lower()) for x in data['lyrics'])


# saving only IDs and lyrics as ELEMENTS_IDS columns
dataset = data[['ID', 'ELEMENTS_IDS']]


# changing the IDs to id_number
def get_ids(x):
	return 'id_' + str(x)
# for num in range(0, len(data['ID'])):
#     print('id_' + str(num))


# defining shingling
def get_my_shing(size, lyric):
	shing_list = set()

	if len(lyric) >= 3:
		for i in range(0, len(lyric)-size+1):
			shing_list.add(abs(hash(''.join(lyric[i:i+size])) % (10 ** 8)))

	else:
		shing_list.add(abs(hash(''.join(lyric[0:len(lyric)])) % (10 ** 8)))
	return shing_list


# getting shingles
dataset['ELEMENTS_IDS'] = [list(get_my_shing(3, dataset['ELEMENTS_IDS'][i].split())) for i in range(0, len(dataset['ELEMENTS_IDS']))]


# changing IDS to id_ID
dataset['ID'] = [(get_ids(dataset['ID'][i])) for i in range(0, len(dataset['ID']))]


# exporting dataset shingle
dataset.to_csv(path_ + "/data/dataset_250k.tsv", sep='\t', index=False)


