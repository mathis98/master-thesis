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

		self.intra = intra

		self.validation_step_outputs = []
		self.test_step_outputs = []

	def forward(self, batch):

		image, caption = batch

		if self.intra:
			copy_img = image
			image = image[0], image[2]
			augmented_image = copy_img[1], copy_img[2]

			copy_caption = caption
			caption = caption[0], caption[2], caption[4]
			augmented_caption = copy_caption[1], copy_caption[3], copy_caption[4]

		image_embed = self.resnet_embedding_module(image)
		image_embed = image_embed.view(image_embed.size(0), -1)
		image_embed = self.projection_head(image_embed)

		caption_embed = self.bert_embedding_module(caption)
		caption_embed = self.projection_head(caption_embed)

		if self.intra:
			augmented_image_embed = self.resnet_embedding_module(augmented_image)
			augmented_image_embed = augmented_image_embed.view(augmented_image_embed.size(0), -1)
			augmented_image_embed = self.projection_head(augmented_image_embed)

			augmented_caption_embed = self.bert_embedding_module(augmented_caption)
			augmented_caption_embed = self.projection_head(augmented_caption_embed)

			return image_embed, augmented_image_embed, caption_embed, augmented_caption_embed
		
		return image_embed, caption_embed

	def training_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption

		if self.intra:
			image_embed, augmented_image_embed, caption_embed, augmented_caption_embed = self(batch)

		else:
			image_embed, caption_embed = self(batch)
		
		loss = self.criterion(image_embed, caption_embed)

		if self.intra:
			intra_image_loss = self.criterion(image_embed, augmented_image_embed)
			intra_caption_loss = self.criterion(caption_embed, augmented_caption_embed)

			loss = loss + intra_image_loss + intra_caption_loss

		self.log('train-loss', loss, prog_bar=True)
		return loss

	def test_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch

		if self.intra:
			image = image[0], image[2]
			caption = caption[0], caption[2], caption[4]

		indeces = caption[2]
		labels = indeces // 100
		groundtruth = relevant_list(labels)

		if self.intra:
			image_embed, augmented_image_embed, caption_embed, augmented_caption_embed = self(batch)

		else:
			image_embed, caption_embed = self(batch)
		
		# image_embed = torch.squeeze(image_embed)
		
		mAP = calculate_mAP(image_embed, caption_embed, groundtruth)
		self.log('train mAP:',np.mean(mAP), batch_size=self.batch_size)
		self.test_step_outputs.append(mAP)

	def on_test_epoch_end(self):
		avg_mAP = np.mean(np.concatenate(self.test_step_outputs))
		self.log('avg_test_mAP: ', avg_mAP, batch_size=self.batch_size, prog_bar=True)
		print('avg_test_mAP: ', avg_mAP)


	def validation_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch

		if self.intra:
			image = image[0], image[2]
			caption = caption[0], caption[2], caption[4]

		indeces = caption[2]
		labels = indeces // 100
		groundtruth = relevant_list(labels)

		if self.intra:
			image_embed, augmented_image_embed, caption_embed, augmented_caption_embed = self(batch)

		else:
			image_embed, caption_embed = self(batch)
		
		mAP = calculate_mAP(image_embed, caption_embed, groundtruth)
		self.log('validation mAP:',np.mean(mAP), batch_size=self.batch_size)
		self.validation_step_outputs.append(mAP)

	def on_validation_epoch_end(self):
		avg_mAP = np.mean(np.concatenate(self.validation_step_outputs))
		self.log('avg_val_mAP: ', avg_mAP, batch_size=self.batch_size, prog_bar=True)
		print('avg_val_mAP: ', avg_mAP)

	def configure_optimizers(self):

		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=self.max_epochs, eta_min=self.learning_rate / 50
		)
		return [optimizer], [lr_scheduler]