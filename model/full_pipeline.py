import sys
sys.path.append('..')

import pytorch_lightning as pl
import torch

# SimCLR for Text
from model.simclr_text_model import SimCLRModule as BertEmbeddingModule

# SimCLR for Image
from model.simclr_model import SimCLRModule as ResNetEmbeddingModel

# SimCLR loss
from loss.contrastive_loss import SimCLRLoss


class FullPipeline(pl.LightningModule):
	def __init__(self, text_data, image_paths, batch_size, temperature=.07):
		super(FullPipeline, self).__init__()
		self.text_data = text_data
		self.image_paths = image_paths
		self.batch_size = batch_size
		self.temperature = temperature
		self.bert_embedding_module = BertEmbeddingModule()
		self.resnet_embedding_module = ResNetEmbeddingModel()
		self.criterion = torch.nn.CrossEntropyLoss()

	def forward(self, images, captions):
		image_embed = self.resnet_embedding_module(images)
		text_embed = self.bert_embedding_module(captions)
		return image_embed, text_embed

	def configure_optimizers(self):
		optimizers = [
			{"params": self.resnet_embedding_module.parameters(), "lr": self.resnet_embedding_module.learning_rate},
			{"params": self.bert_embedding_module.parameters(), "lr": self.bert_embedding_module.learning_rate}
		]

		optimizer = torch.optim.Adam(optimizers)
		return optimizer

	def training_step(self, batch, batch_idx, dataloader_idx):
		print(batch)


		# contrastive loss between image and caption


		# simclr loss between image and image


		# simclr loss between text and text


		# calculate total loss and return

	def validation_step(self, batch, batch_idx, dataloader_idx):
		pass