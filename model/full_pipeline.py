import sys
sys.path.append('..')

import pytorch_lightning as pl
import torch
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
	def __init__(self, batch_size=64, temperature=.07, learning_rate=1e-4):
		super(FullPipeline, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.learning_rate = learning_rate

		self.bert_embedding_module = BERTSentenceEmbedding()
		self.resnet_embedding_module = ImageEmbeddingModule()
		self.criterion = SimCLRLoss(temperature)

		self.validation_step_outputs = []
		self.test_step_outputs = []

	def forward(self, image, caption):
		image_embed = self.resnet_embedding_module(image)
		text_embed = self.bert_embedding_module(caption)
		return image_embed, text_embed

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer

	def training_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch
		image_embed, caption_embed = self(image, caption)
		image_embed = torch.squeeze(image_embed)
		loss = self.criterion(image_embed, caption_embed)
		self.log('train-loss', loss, batch_size=self.batch_size)
		return loss

	def test_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch

		indeces = caption[2]
		labels = indeces // 100
		groundtruth = relevant_list(labels)

		image_embed, caption_embed = self(image, caption)
		image_embed = torch.squeeze(image_embed)
		
		mAP = calculate_mAP(image_embed, caption_embed, groundtruth)
		self.log('train mAP:',mAP)
		self.test_step_outputs.append(mAP)
		return mAP

	def on_test_epoch_end(self):
		avg_mAP = np.mean(self.test_step_outputs)
		self.log('avg_test_mAP: ', avg_mAP, batch_size=self.batch_size)
		print('avg_test_mAP: ', avg_mAP)


	def validation_step(self, batch, batch_idx):
		# NT-Xent loss between image and caption
		image, caption = batch

		indeces = caption[2]
		labels = indeces // 100
		groundtruth = relevant_list(labels)

		image_embed, caption_embed = self(image, caption)
		image_embed = torch.squeeze(image_embed)
		
		mAP = calculate_mAP(image_embed, caption_embed, groundtruth)
		self.log('validation mAP:',mAP)
		self.validation_step_outputs.append(mAP)
		return mAP

	def on_validation_epoch_end(self):
		avg_mAP = np.mean(self.validation_step_outputs)
		self.log('avg_val_mAP: ', avg_mAP, batch_size=self.batch_size)
		print('avg_val_mAP: ', avg_mAP)