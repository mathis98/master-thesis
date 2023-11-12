import sys
sys.path.append('..')

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np

# Embedding for text
from model.text_embedding import BERTSentenceEmbedding

# Embedding for image
from model.image_embedding import ImageEmbeddingModule

# SimCLR loss
from loss.contrastive_loss import SimCLRLoss

# for mAP calculation
from utility.helpers import relevant_list, calculate_mAP


class FullPipeline(pl.LightningModule):
	def __init__(self, batch_size=64, intra=False, temperature=.07, learning_rate=1e-4, weight_decay=1e-4, max_epochs=500, hidden_dim=128):
		super(FullPipeline, self).__init__()
		self.batch_size = batch_size
		self.intra = intra
		self.temperature = temperature
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.hidden_dim = hidden_dim

		self.resnet_embedding_module = ImageEmbeddingModule()
		self.bert_embedding_module = BERTSentenceEmbedding()

		self.projection_head = nn.Sequential(
			nn.Linear(512, 4*hidden_dim),
			nn.ReLU(),
			nn.Linear(4*hidden_dim, hidden_dim)
		)
		
		self.criterion = SimCLRLoss(temperature)
		self.max_epochs = max_epochs

		self.validation_step_outputs = []
		self.test_step_outputs = []

	def forward(self, batch):

		image, caption = batch

		image_embed = self.resnet_embedding_module(image)
		image_embed = image_embed.view(image_embed.size(0), -1)
		image_embed = self.projection_head(image_embed)

		text_embed = self.bert_embedding_module(caption)
		text_embed = self.projection_head(text_embed)
		
		return image_embed, text_embed

	def training_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption

		image_embed, caption_embed = self(batch)
		
		# image_embed = torch.squeeze(image_embed)
		
		loss = self.criterion(image_embed, caption_embed)

		self.log('train-loss', loss, prog_bar=True)
		return loss

	def test_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch

		indeces = caption[2]
		labels = indeces // 100
		groundtruth = relevant_list(labels)

		image_embed, caption_embed = self(batch)
		
		# image_embed = torch.squeeze(image_embed)
		
		mAP = calculate_mAP(image_embed, caption_embed, groundtruth)
		self.log('train mAP:',np.mean(mAP), batch_size=self.batch_size)
		self.test_step_outputs.append(mAP)

	def on_test_epoch_end(self):
		avg_mAP = np.mean([output for output in self.test_step_outputs])
		self.log('avg_test_mAP: ', avg_mAP, batch_size=self.batch_size, prog_bar=True)
		print('avg_test_mAP: ', avg_mAP)


	def validation_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch

		indeces = caption[2]
		labels = indeces // 100
		groundtruth = relevant_list(labels)

		image_embed, caption_embed = self(batch)
	
		# image_embed = torch.squeeze(image_embed)
		
		mAP = calculate_mAP(image_embed, caption_embed, groundtruth)
		self.log('validation mAP:',np.mean(mAP), batch_size=self.batch_size)
		self.validation_step_outputs.append(mAP)

	def on_validation_epoch_end(self):
		print('validation step outputs: ', self.validation_step_outputs)
		self.validation_step_outputs = [output for output in self.validation_step_outputs]
		avg_mAP = np.mean(self.validation_step_outputs)
		self.log('avg_val_mAP: ', avg_mAP, batch_size=self.batch_size, prog_bar=True)
		print('avg_val_mAP: ', avg_mAP)

	def configure_optimizers(self):

		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=self.max_epochs, eta_min=self.learning_rate / 50
		)
		return [optimizer], [lr_scheduler]