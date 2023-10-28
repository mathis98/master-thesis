import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from argument_parser import parse_arguments
from util import closest_indices

args = parse_arguments()


class CustomSentenceDataset(Dataset):
	def __init__(self, json_file_path, tokenizer, max_length=128):
		
		with open(json_file_path, 'r') as json_file:
			data = json.load(json_file)

		self.sentences = [item['sentences'][0]['raw'] for item in data['images']]

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
		self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
		self.dataset = CustomSentenceDataset(self.json_file_path, self.tokenizer)

	def train_dataloader(self):	
		return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


class BERTSentenceEmbedding(pl.LightningModule):
	def __init__(self, model_name, embedding):
		super(BERTSentenceEmbedding, self).__init__()
		self.model_name = model_name
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.embedding = embedding

		if self.embedding == 'sbert':
			self.model = SentenceTransformer('all-mpnet-base-v2')

		else:
			self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

	def forward(self, inputs):
		if self.embedding != 'sbert':
			outputs = self.model(inputs[0]['input_ids'], inputs[0]['attention_mask'])

		if self.embedding == 'CLS':
			return outputs.last_hidden_state[:, 0, :]

		elif self.embedding == 'last':
			last_hidden = outputs.last_hidden_state
			return torch.mean(last_hidden, dim=1) 

		elif self.embedding == 'last_n':
			last_n_hidden = outputs.hidden_states[-4:]
			hiddens = torch.stack(last_n_hidden)
			resulting_states = torch.sum(hiddens, dim=0)

			return torch.mean(resulting_states, dim=1)

		elif self.embedding == 'sbert':
			return self.model.encode(inputs[1])



model_name = 'prajjwal1/bert-mini'
batch_size = 64

path = '../Datasets/UCM/dataset.json' if not args.ucm else '../Datasets/RSICD/dataset_rsicd.json'

print(f"Using {path}")
print(f"Using {args.embedding}")

data_module = SentenceDataModule(model_name, batch_size, path)
data_module.prepare_data()
data_module.setup()

bert_embedding = BERTSentenceEmbedding(model_name, args.embedding)

summary(bert_embedding)

trainer = pl.Trainer()
embeddings = []

with torch.no_grad():
	predictions = trainer.predict(bert_embedding, data_module.train_dataloader())


for batch in predictions:
	embeddings.extend(batch.tolist())

embeddings = np.vstack(embeddings)

print(f"Shape: {embeddings.shape}")

pairs = closest_indices(embeddings)

for first, second in pairs:
	print(data_module.dataset.sentences[first], '<-->', data_module.dataset.sentences[second], ' (', first, ',', second, ')')