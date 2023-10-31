import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json

import nlpaug.augmenter.word as naw

class SimCLRDataset(Dataset):
	def __init__(self, json_file_path, tokenizer, max_length=128):
		
		with open(json_file_path, 'r') as json_file:
			data = json.load(json_file)

		self.sentences = [item['sentences'][0]['raw'] for item in data['images']]

		self.tokenizer = tokenizer
		self.max_length = max_length
		self.aug = naw.SynonymAug(aug_src='wordnet')

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

		return inputs, inputs_aug, sentence, aug_sentence, idx


class SimCLRDataModule(pl.LightningDataModule):
	def __init__(self, model_name, batch_size, json_file_path):
		super(SimCLRDataModule, self).__init__()
		self.model_name = model_name
		self.batch_size = batch_size
		self.json_file_path = json_file_path 

	def setup(self, stage=None):
		self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
		self.dataset = SimCLRDataset(self.json_file_path, self.tokenizer)

	def train_dataloader(self):	
		return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
