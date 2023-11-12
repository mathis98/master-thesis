import torch
import torch.nn as nn
import pytorch_lightning as pl

import torch
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer


import sys
sys.path.append('..')

from loss.contrastive_loss import SimCLRLoss


class SimCLRModule(pl.LightningModule):
	def __init__(self, model_name='prajjwal1/bert-small', embedding='CLS', temperature=.07, learning_rate=1e-4, hidden_dim=128):
		super(SimCLRModule, self).__init__()
		self.model_name = model_name
		self.embedding = embedding
		self.temperature = temperature
		self.criterion = SimCLRLoss(temperature)
		self.learning_rate = learning_rate


		if self.embedding == 'sbert':
			self.model = SentenceTransformer('all-mpnet-base-v2')

		else:
			self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
			self.projection_head = nn.Sequential(
				nn.Linear(self.model.config.hidden_size, 4*hidden_dim),
				nn.ReLU(),
				nn.Linear(4*hidden_dim, hidden_dim)
			)

	def forward(self, batch):

		original, augmented, _, _, _ = batch

		if self.embedding != 'sbert':
			outputs = self.model(original['input_ids'], original['attention_mask'])
			outputs_aug = self.model(augmented['input_ids'], augmented['attention_mask'])

		if self.embedding == 'CLS':
			outputs, outputs_aug =  outputs.last_hidden_state[:, 0, :], outputs_aug.last_hidden_state[:, 0, :]
			return self.projection_head(outputs), self.projection_head(outputs_aug)

		elif self.embedding == 'last':
			last_hidden = outputs.last_hidden_state
			last_hidden_aug = outputs_aug.last_hidden_state
			return torch.mean(last_hidden, dim=1), torch.mean(last_hidden_aug, dim=1)

		elif self.embedding == 'last_n':
			last_n_hidden = outputs.hidden_states[-4:]
			hiddens = torch.stack(last_n_hidden)
			resulting_states = torch.sum(hiddens, dim=0)

			last_n_hidden_aug = outputs_aug.hidden_state[-4:]
			hiddens_aug = torch.stack(last_n_hidden_aug)
			resulting_states_aug = torch.sum(hiddens_aug, dim=0)

			return torch.mean(resulting_states, dim=1), torch.mean(resulting_states_aug, dim=1)

		elif self.embedding == 'sbert':
			return self.model.encode(original[2]), self.model.encode(augmented[3])


	def training_step(self, batch, batch_idx):
		z_original, z_augmented = self(batch)
		loss = self.criterion(z_original, z_augmented)
		self.log("train_loss", loss, prog_bar=True)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer