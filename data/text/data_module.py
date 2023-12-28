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


	def __getitem__(self,idx):
		sentence = self.sentences[idx]

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

		return inputs, sentence, self.indices[idx]


class SentenceDataModule(pl.LightningDataModule):
	"""
	A Data Module for captions

	Args:
		model_name (str): BERT model to use for embedding
		batch_size (int): batch size for data loader
		json_file_path (str): path to the json file containing the captions
		seed (int): seed for shuffling
	"""
	def __init__(self, model_name, batch_size, json_file_path, seed=42, num_repeats=5, technique='Concat', rand=3):
		super(SentenceDataModule, self).__init__()
		self.model_name = model_name
		self.batch_size = batch_size
		self.json_file_path = json_file_path 
		self.seed = seed
		self.num_repeats = num_repeats
		self.technique = technique
		self.rand = rand

	def setup(self, stage=None):
		self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

		with open(self.json_file_path, 'r') as json_file:
			data = json.load(json_file)

		if self.technique == 'Concat':

			if 'NWPU' in self.json_file_path:
				sentences = []
				categories = [category for category in data]

				for category in categories:
					sentences.extend([' '.join([item[f'raw_{i}'] for i in range(1,5)]) for item in data[category]])
				print(sentences[:5])
			else:
				sentences = [' '.join([item['sentences'][i]['raw'] for i in range(5)]) for item in data['images']]

		elif self.technique == 'Random':

			if 'NWPU' in self.json_file_path:
				pass
			else:
				sentences = [[item['sentences'][i]['raw'] for i in range(5)][self.rand] for item in data['images']]

		elif self.technique == 'Repeat':
			if 'NWPU' in self.json_file_path:
				pass
			else:
				sentences = [[item['sentences'][i]['raw'] for i in range(5)] for item in data['images']]
				sentences = list(itertools.chain.from_iterable(sentences))

		print(f'Using {self.technique}')

		total_size = len(sentences)
		train_size = int(.8 * total_size)
		val_size = int(.1 * total_size)
		test_size = total_size - train_size - val_size

		indices = list(range(total_size))

		np.random.seed(self.seed)
		shuffled_indices = np.random.permutation(indices)

		train_indices = []
		val_indices = []
		test_indices = []

		elements_per_group = 100 * self.num_repeats

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

		self.dataset = CustomSentenceDataset(sentences, self.tokenizer, indices)

		self.train_dataset = CustomSentenceDataset([sentences[i] for i in train_indices], self.tokenizer, train_indices)
		self.val_dataset = CustomSentenceDataset([sentences[i] for i in val_indices], self.tokenizer, val_indices)
		self.test_dataset = CustomSentenceDataset([sentences[i] for i in test_indices], self.tokenizer, test_indices)

	def train_dataloader(self):	
		return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=30)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=30)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=30)
