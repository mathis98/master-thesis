import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
import numpy as np
import json
import itertools

import nlpaug.augmenter.word as naw

class SimCLRDataset(Dataset):
	def __init__(self, sentences, tokenizer, indices, max_length=128):
		
		self.sentences = sentences
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.aug = naw.SynonymAug(aug_src='wordnet')
		self.indices = indices

	def __len__(self):
		return len(self.sentences)


	def __getitem__(self,idx):
		sentence = self.sentences[idx]
		aug_sentence = self.aug.augment(sentence)[0]

		inputs = self.tokenizer.encode_plus(
			sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            max_length=self.max_length
			)

		inputs_aug = self.tokenizer.encode_plus(
			aug_sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            max_length=self.max_length
			)

		inputs['input_ids'] = inputs['input_ids'].squeeze(0)
		inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)

		inputs_aug['input_ids'] = inputs['input_ids'].squeeze(0)
		inputs_aug['attention_mask'] = inputs['attention_mask'].squeeze(0)

		return inputs, inputs_aug, sentence, aug_sentence, self.indices[idx]


class SimCLRDataModule(pl.LightningDataModule):
	def __init__(self, batch_size, json_file_path, tokenizer, seed=42):
		super(SimCLRDataModule, self).__init__()
		self.batch_size = batch_size
		self.json_file_path = json_file_path 
		self.seed = seed
		self.tokenizer = tokenizer

	def setup(self, stage=None):

		with open(self.json_file_path, 'r') as json_file:
			data = json.load(json_file)

		sentences = [[item['sentences'][i]['raw'] for i in range(5)] for item in data['images']]
		sentences = list(itertools.chain.from_iterable(sentences))

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

		elements_per_group = 100

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

		self.dataset = SimCLRDataset([sentences[i] for i in shuffled_indices], self.tokenizer, shuffled_indices)

		self.train_dataset = SimCLRDataset([sentences[i] for i in train_indices], self.tokenizer, train_indices)
		self.val_dataset = SimCLRDataset([sentences[i] for i in val_indices], self.tokenizer, val_indices)
		self.test_dataset = SimCLRDataset([sentences[i] for i in test_indices], self.tokenizer, test_indices)

	def train_dataloader(self):	
		return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=30)
