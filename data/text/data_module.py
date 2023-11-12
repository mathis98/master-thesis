import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import itertools


class CustomSentenceDataset(Dataset):
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
	def __init__(self, model_name, batch_size, json_file_path, seed=42):
		super(SentenceDataModule, self).__init__()
		self.model_name = model_name
		self.batch_size = batch_size
		self.json_file_path = json_file_path 
		self.seed = seed

	def setup(self, stage=None):
		self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

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

		# print('sentences:')
		# print(sentences[-10:])

		# 5 captions per image: [0,100] -> [0,500], [3, 20] -> [11, 100]

		# def get_sentences(indeces):
		# 	items_filter = []
		# 	for index in indeces:
		# 		items_filter.append(items[index])
		# 	all_sentences = [[item['sentences'][i]['raw'] for i in range(5)] for item in items_filter]
		# 	return list(itertools.chain.from_iterable(all_sentences))

		train_indices, val_indices, test_indices = shuffled_indices[:train_size], shuffled_indices[train_size:(train_size+val_size)], shuffled_indices[(train_size+val_size):]

		self.dataset = CustomSentenceDataset([sentences[i] for i in shuffled_indices], self.tokenizer, shuffled_indices)

		self.train_dataset = CustomSentenceDataset([sentences[i] for i in train_indices], self.tokenizer, train_indices)
		self.val_dataset = CustomSentenceDataset([sentences[i] for i in val_indices], self.tokenizer, val_indices)
		self.test_dataset = CustomSentenceDataset([sentences[i] for i in test_indices], self.tokenizer, test_indices)

	def train_dataloader(self):	
		return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=30)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=30)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=30)
