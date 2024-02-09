import json
import os

json_file_path = '../../Datasets/NWPU-Captions-main/dataset_nwpu.json'

with open(json_file_path, 'r') as json_file:
	data = json.load(json_file)

sentences = []
categories = sorted([category for category in data])

weird_ones = []

for category in categories:

	words = category.split('_')

	for idx, item in enumerate(data[category]):

		weird = True

		sentences = [item['raw']] + [item[f'raw_{i}'] for i in range(1, 5)]

		for sentence in sentences:
			for word in words:
				if word in sentence:
					weird = False
		if weird: 
			weird_ones.append(f'{category}_{idx}')

print(f'WEIRD: {weird_ones}')
