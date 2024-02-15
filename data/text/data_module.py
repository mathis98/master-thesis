import os
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import itertools


class CustomSentenceDataset(Dataset):
	"""
	Dataset class for caption embeddings.

	Args:
		sentences (list): List of captions.
		tokenizer: Tokenizer for encoding captions.
		indices (list): List of indices of captions.
		max_length (int): Maximum length of tokenized caption.
	"""

	def __init__(self, sentences, tokenizer, indices, max_length=128):

		self.sentences = sentences
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.indices = indices

	def __len__(self):
		return len(self.sentences)


	# RETURN Uniqueness scores if technique = 'Info'
	def __getitem__(self,idx):
		sentences = self.sentences[idx]

		# multiple sentences return as list
		if isinstance(sentences, list):
			sentence_list = []
			for sentence in sentences:

				inputs = self.tokenizer.encode_plus(
					sentence,
		            return_tensors="pt",
		            padding="max_length",
		            truncation=True,
		            return_attention_mask=True,
		            return_token_type_ids=False,
		            max_length=self.max_length
				)

				inputs['input_ids'] = inputs['input_ids'].squeeze(0)
				inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)

				sentence_list.append((inputs, sentence, self.indices[idx]))
			return sentence_list

		# single sentence
		inputs = self.tokenizer.encode_plus(
			sentences,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            max_length=self.max_length
			)

		inputs['input_ids'] = inputs['input_ids'].squeeze(0)
		inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)

		# Return tokenied sentence, raw sentence, and index of sentence in the dataset
		return inputs, sentences, self.indices[idx]


class SentenceDataModule(pl.LightningDataModule):
	"""
	A Data Module for captions

	Args:
		model_name (str): BERT model to use for embedding
		batch_size (int): batch size for data loader
		json_file_path (str): path to the json file containing the captions
		seed (int): seed for shuffling
	"""
	def __init__(self, model_name, batch_size, json_file_path, seed=42, num_repeats=5, technique='Repeat', rand=3):
		super(SentenceDataModule, self).__init__()
		self.model_name = model_name
		self.batch_size = batch_size
		self.json_file_path = json_file_path 
		self.seed = seed
		self.num_repeats = num_repeats
		self.technique = technique


		np.random.seed(888)

		self.rand = 4

	def setup(self, stage=None):
		self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

		# Open the JSON file containing the image caption mapping
		with open(self.json_file_path, 'r') as json_file:
			data = json.load(json_file)

		# For the Concatenation Technique
		if self.technique == 'Concat':

			# NWPU Dataset
			if 'NWPU' in self.json_file_path:
				# Go through each item and concatenate raw, raw_1, raw_2, raw_3, and raw_4
				sentences = []
				categories = sorted([category for category in data])

				for category in categories:
					sentences.extend([' '.join([item['raw']] + [item[f'raw_{i}'] for i in range(1, 5)]) for item in data[category]])
			# UCM dataset
			else:
				# Go through each item and concatentate 'raw' for 'sentences'[1-4]
				sentences = [' '.join([item['sentences'][i]['raw'] for i in range(5)]) for item in data['images']]

		# For the Random Technique
		elif self.technique == 'Random':

			# NWPU Dataset
			if 'NWPU' in self.json_file_path:
				# Go through each item and store raw_{rand} where rand is passed in
				sentences = []
				categories = sorted([category for category in data])

				for category in categories:
					key = 'raw' if self.rand == 0 else f'raw_{self.rand}'
					sentences.extend([item[key] for item in data[category]])
			# UCM Dataset
			else:
				# For each item select 'raw' from 'sentences'[{rand}] where rand is passed in
 				sentences = [[item['sentences'][i]['raw'] for i in range(5)][self.rand] for item in data['images']]

 		# For the Repeat Technique
		elif self.technique == 'Repeat':

			# NWPU Dataset
			if 'NWPU' in self.json_file_path:
				# For each item keep all captions (raw, raw_1, raw_2, raw_3, raw_4), flatten the list
				sentences = []
				categories = sorted([category for category in data])

				for category in categories:
					sentences.extend([item['raw']] + [item[f'raw_{i}'] for i in range(1, 5)] for item in data[category])
				sentences = list(itertools.chain.from_iterable(sentences))
			# UCM dataset
			else:
				# For each item go through 'sentences'[0-4] and keep all captions, flatten the list
				sentences = [[item['sentences'][i]['raw'] for i in range(5)] for item in data['images']]
				sentences = list(itertools.chain.from_iterable(sentences))

		# Mean, RankAgg, Info, Learned_FC, Learned_Att
		# FOR Info ===> CALCULATE Uniqueness Score additionally and store WITH THE SENTENCES!!
		# Store them in a separate data structure
		elif self.technique in ['Mean', 'RankAgg', 'Info', 'Learned_FC', 'Learned_Att']:
			# Mean Feature technique, also for Rank Aggregation
			# ==> List [[caption1_1, caption2_1, caption_3_1, caption4_1, caption5_1],[caption1_2, caption2_2,...],...]
			
			# NWPU dataset
			if 'NWPU' in self.json_file_path:
				# For each item keep all captions (raw, raw_1, raw_2, raw_3, raw_4), do NOT flatten
				sentences = []
				categories = sorted([category for category in data])

				for category in categories:
					sentences.extend([item['raw']] + [item[f'raw_{i}'] for i in range(1, 5)] for item in data[category])
			# UCM dataset
			else:
				# For each item go through 'sentences'[0-4] and keep all captions, do NOT flatten
				sentences = [[item['sentences'][i]['raw'] for i in range(5)] for item in data['images']]


		# Total number of captions we have:
		# Concat, Random: Same as images
		# Repeat: num_repeats * images (actually same as images are repeated)
		total_size = len(sentences)

		# Calculate train, test, val sizes (80%, 10%, 10%)
		train_size = int(.8 * total_size)
		val_size = int(.1 * total_size)
		test_size = total_size - train_size - val_size

		# Construct indices from 0 to total_size-1
		indices = list(range(total_size))

		np.random.seed(self.seed)

		train_indices = []
		val_indices = []
		test_indices = []

		# 100 elements per class for UCM, 700 for NWPU
		elements_per_group = 100 * self.num_repeats

		if 'NWPU' in self.json_file_path:
			elements_per_group = 700 * self.num_repeats

		# Iterate through each group
		for group_start in range(0, len(indices), elements_per_group):
			group_end = group_start + elements_per_group
			group = indices[group_start:group_end]

			# Calculate the indices for train, val, and test
			train_end = int(len(group) * 0.8)
			val_end = train_end + int(len(group) * 0.1)

			# Split the group into train, val, and test
			train_indices.extend(group[:train_end])
			val_indices.extend(group[train_end:val_end])
			test_indices.extend(group[val_end:])

		# Construct complete dataset with all captions
		self.dataset = CustomSentenceDataset(sentences, self.tokenizer, indices)

		# And sub datasets for training, validation, and testing
		self.train_dataset = CustomSentenceDataset([sentences[i] for i in train_indices], self.tokenizer, train_indices)
		self.val_dataset = CustomSentenceDataset([sentences[i] for i in val_indices], self.tokenizer, val_indices)
		self.test_dataset = CustomSentenceDataset([sentences[i] for i in test_indices], self.tokenizer, test_indices)

	# Dataloaders for training, validation, and testing
	def train_dataloader(self):	
		return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=30)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=30)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=30)
