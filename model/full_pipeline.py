import sys
sys.path.append('..')

import pytorch_lightning as pl
import torch

# Embedding for text
from model.text_embedding import BERTSentenceEmbedding

# Embedding for image
from model.image_embedding import ImageEmbeddingModule

# SimCLR loss
from loss.contrastive_loss import SimCLRLoss


class FullPipeline(pl.LightningModule):
	def __init__(self, batch_size=64, temperature=.07, learning_rate=1e-4):
		super(FullPipeline, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.learning_rate = learning_rate

		self.bert_embedding_module = BERTSentenceEmbedding()
		self.resnet_embedding_module = ImageEmbeddingModule()
		self.criterion = SimCLRLoss(temperature)

	def forward(self, image, caption):
		image_embed = self.resnet_embedding_module(image)
		text_embed = self.bert_embedding_module(caption)
		return image_embed, text_embed

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)

		return {'optimizer': optimizer, 'lr_scheduler': scheduler}

	def training_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch
		caption_embed, img_embed = self(image, caption)
		caption_embed = caption_embed.view(64,512)
		loss = self.criterion(caption_embed, img_embed)
		self.log('train-loss', loss)
		return loss

	def test_step(self, batch, batch_idx):

		# NT-Xent loss between image and caption
		image, caption = batch
		caption_embed, img_embed = self(image, caption)
		caption_embed = caption_embed.view(64,512)
		loss = self.criterion(caption_embed, img_embed)
		self.log('test_loss', loss)
		return loss

	def validation_step(self, batch, batch_idx):
		
		# NT-Xent loss between image and caption
		image, caption = batch
		caption_embed, img_embed = self(image, caption)
		caption_embed = caption_embed.view(64,512)
		loss = self.criterion(caption_embed, img_embed)
		self.log('val_loss', loss)
		return loss