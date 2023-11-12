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

		images = [item for item in data['images']]

		print('count images: ', len(images))

		sentences = []

		for image in images:
			sents = image['sentences']
			for idx in range(5):
				sentences.append(sents[idx]['raw'])

		# sentences = [[item['sentences'][i]['raw'] for i in range(5)] for item in data['images']]
		# sentences = list(itertools.chain.from_iterable(sentences))

		total_size = len(sentences)
		indices = list(range(total_size))

		print('total size: ', total_size)

		np.random.seed(self.seed)
		shuffled_indices = np.random.permutation(indices)

		self.train_dataset = SimCLRDataset([sentences[i] for i in shuffled_indices], self.tokenizer, shuffled_indices)

	def train_dataloader(self):	
		return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=30)
