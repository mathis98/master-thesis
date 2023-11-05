import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import itertools


class CustomSentenceDataset(Dataset):
	def __init__(self, sentences, tokenizer, max_length=128):

		self.sentences = sentences
		self.tokenizer = tokenizer
		self.max_length = max_length

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

		return inputs, sentence, idx


class SentenceDataModule(pl.LightningDataModule):
	def __init__(self, model_name, batch_size, json_file_path):
		super(SentenceDataModule, self).__init__()
		self.model_name = model_name
		self.batch_size = batch_size
		self.json_file_path = json_file_path 

	def setup(self, stage=None):
		self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

		with open(self.json_file_path, 'r') as json_file:
			data = json.load(json_file)

		items = [item for item in data['images']]

		total_size = len(items)
		train_size = int(.8 * total_size)
		val_size = int(.1 * total_size)
		test_size = total_size - train_size - val_size

		indices = list(range(total_size))

		sentences = [[item['sentences'][i]['raw'] for i in range(5)] for item in data['images']]
		sentences = list(itertools.chain.from_iterable(sentences))
		# print('sentences:')
		# print(sentences[0:10])

		# 5 captions per image: [0,100] -> [0,500], [3, 20] -> [11, 100]

		def get_sentences(indeces):
			items_filter = []
			for index in indeces:
				items_filter.append(items[index])
			all_sentences = [[item['sentences'][i]['raw'] for i in range(5)] for item in items_filter]
			return list(itertools.chain.from_iterable(all_sentences))

		train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:(train_size+val_size)], indices[(train_size+val_size):]

		self.train_dataset = CustomSentenceDataset(get_sentences(train_indices), self.tokenizer)
		self.val_dataset = CustomSentenceDataset(get_sentences(val_indices), self.tokenizer)
		self.test_dataset = CustomSentenceDataset(get_sentences(test_indices), self.tokenizer)

	def train_dataloader(self):	
		return DataLoader(self.train_dataset, batch_size=self.batch_size)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size)
