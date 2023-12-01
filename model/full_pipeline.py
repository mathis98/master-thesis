import sys
sys.path.append('..')

import lightning.pytorch as pl
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Embedding for text
from model.text_embedding import BERTSentenceEmbedding

# Embedding for image
from model.image_embedding import ImageEmbeddingModule

# Projection Head
from model.projection_head import MyProjectionhead

# SimCLR loss
from loss.contrastive_loss import SimCLRLoss
# from lightly.loss import NTXentLoss

# for mAP calculation
from utility.helpers import relevant_list, calculate_mAP, define_param_groups, to_cuda_recursive


class FullPipeline(pl.LightningModule):
	"""
	Full Pipeline using Pytorch Lightning with a Text classifier, image classifier, and projection head

	Args:
		batch_size (int): batch size for train, val, and test.
		intra (bool): whether to use intra modal loss.
		temperature (float): ntxent temperature.
		learning_rate (float): AdamW learning rate.
		weight_decay (float): AdamW weight decay (L2 regularization).
		max_epochs (int): maximum epochs to train for.
		hidden_dim (int): dimension of embedding.

	Attributes:
		batch_size (int): batch size for train, val, and test.
		intra (bool): whether to use intra modal loss.
		temperature (float): ntxent temperature.
		learning_rate (float): AdamW learning rate.
		weight_decay (float): AdamW weight decay (L2 regularization).
		hidden_dim (int): dimension of embedding.
		resnet_embedding_module (model.image_embedding.ImageEmbeddingModule): Image embedding module.
		bert_embedding_module (model.text_embedding.BERTSentenceEmbeddingModule): Caption embedding module.
		projection_head (model.projection_head.MyProjectionHead): Projection head to shared embedding space.
		criterion (loss.contrastive_loss.SimCLRLoss): NT-Xent loss function.
		max_epochs (int): Maximum epochs to train for.
		validation_step_outputs (list): List to store mAP values during validation.
		test_step_outputs (list): List to store mAP values during testing.
		val_dataloader (DataLoader): Dataloader for validation set.
		test_dataloader (DataLoader): Dataloader for test set.

	Methods:
		forward(batch): Forward pass through the model.
		training_step(batch, batch_idx): Training step.
		shared_step(batch): Shared step for both validation and testing.
		test_step(batch, batch_idx): Test step.
		on_test_epoch_end(): Called at the end of the test epoch to calculate and log avg mAP.
		validation_step(batch, batch_idx): Validation step.
		on_validation_epoch_end(): Called at the end of the validation epoch to calculate and log avg mAP.
		configure_optimizers(): Configure the optimizer.
		val_dataloader (DataLoader): Dataloader for validation set.
		test_dataloader (DataLoader): Dataloader for test set.
	"""
	def __init__(self, val_dataloader, test_dataloader, batch_size=128, intra=False, temperature=.5, learning_rate=1e-4, weight_decay=1e-6, max_epochs=100, hidden_dim=128):
		super(FullPipeline, self).__init__()
		self.batch_size = batch_size
		self.intra = intra
		self.temperature = temperature
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.hidden_dim = hidden_dim

		self.resnet_embedding_module = ImageEmbeddingModule()
		# Freeze weights
		# for param in self.resnet_embedding_module.parameters():
		# 	param.requires_grad = False


		self.bert_embedding_module = BERTSentenceEmbedding()
		# Freeze weights
		for param in self.bert_embedding_module.parameters():
			param.requires_grad = False

		self.projection_head = MyProjectionhead(512, 512, 128)
		
		self.criterion = SimCLRLoss(temperature)
		# self.criterion = NTXentLoss(temperature)
		self.max_epochs = max_epochs

		# To calculate all image embeddings at start of val and test epochs
		self.validation_embeddings = []
		self.test_embeddings = []

		self.validation_step_outputs = []
		self.test_step_outputs = []

		self.val_dataloader = val_dataloader
		self.test_dataloader = test_dataloader

	def forward(self, batch):
		"""
		Forward pass through the model.

		Args:
			batch: Input batch with image, caption pairs.

		Returns:
			Tuple: Embeddings of the input pairs.
		"""

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
		"""
		Training step.

		Args:
			batch: Input batch for the training.
			batch_idx: Index of the current batch.

		Returns:
			torch.Tensor: NT-Xent loss inter modal (+optional intra modal)
		"""

		# NT-Xent loss between image and caption

		if self.intra:
			image_embed, augmented_image_embed, caption_embed, augmented_caption_embed = self(batch)

		else:
			image_embed, caption_embed = self(batch)

		image_embed = F.normalize(image_embed, dim=-1, p=2)
		caption_embed = F.normalize(caption_embed, dim=-1, p=2)
		
		loss = self.criterion(image_embed, caption_embed)

		if self.intra:
			intra_image_loss = self.criterion(image_embed, augmented_image_embed)
			intra_caption_loss = self.criterion(caption_embed, augmented_caption_embed)

			loss = loss + intra_image_loss + intra_caption_loss

		self.log('train-loss', loss, prog_bar=True)
		return loss

	def calculate_embeddings_for_images(self, validation=True):
		"""
		Calculate and return embeddings for the entire set of images.

		Args:
			validation (bool): Whether to calculate embeddings for the validation or test set.

		Returns:
			List: Embeddings for the entire set of images.
		"""

		# Get the appropriate DataLoader
		dataloader = self.val_dataloader() if validation else self.test_dataloader()

		# List to store embeddings
		image_embeddings = []

		# Set to evaluation mode
		self.eval()

		# Offers speedup, don't calculate gradients
		with torch.no_grad():		
			for batch in dataloader:
				# Forward pass to get image embeddings
				if self.intra:
					image_embed, _, _, _ = self(to_cuda_recursive(batch))

				else:
					image_embed, _ = self(to_cuda_recursive(batch))

				image_embed = F.normalize(image_embed, dim=-1, p=2)
				image_embeddings.append(image_embed.detach().cpu().numpy())

		# Concatenate embeddings
		image_embeddings = np.concatenate(image_embeddings)

		return image_embeddings


	def shared_step(self, batch, validation=True):
		"""
		Shared step for both validation and test steps.

		Args:
			batch: Input batch for validation or test with image, caption pairs.

		Returns:
			np.array: mAP (mean Average Precision) values for input batch, based on embeddings.
		"""

		image, caption = batch

		if self.intra:
			image = image[0], image[2]
			caption = caption[0], caption[2], caption[4]

		indeces = caption[2]
		labels = indeces // 500 # They need to be same if in same class
		groundtruth = relevant_list(labels)

		if self.intra:
			image_embed, _, caption_embed, _ = self(batch)

		else:
			image_embed, caption_embed = self(batch)

		# image_embed = F.normalize(image_embed, dim=-1, p=2)
		caption_embed = F.normalize(caption_embed, dim=-1, p=2)

		image_embeddings = self.validation_embeddings if validation else self.test_embeddings

		mAP = calculate_mAP(image_embeddings, caption_embed, groundtruth)

		return mAP


	def on_test_epoch_start(self):
		"""
		Called at the beginning of the test epoch.
		"""

		# Calculate and store embeddings for the entire test set.
		self.test_embeddings = self.calculate_embeddings_for_images(validation=False)


	def test_step(self, batch, batch_idx):
		"""
		Test step.

		Args:
			batch: Input batch for testing. Image, caption pairs.
			batch_idx: Index of the current batch.
		"""

		mAP = self.shared_step(batch)
		self.log('train mAP',np.mean(mAP), batch_size=self.batch_size)
		self.test_step_outputs.append(mAP)

	def on_test_epoch_end(self):
		"""
		Called at the end of the test epoch to calculate and log the average mAP.
		"""
		avg_mAP = np.mean(np.concatenate(self.test_step_outputs))
		self.log('avg_test_mAP', avg_mAP, batch_size=self.batch_size, prog_bar=True)


	def on_validation_epoch_start(self):
		"""
		Called at the beginning of the validation epoch.
		"""

		# Calculate and store embeddings for the entire validation set.
		self.validation_embeddings = self.calculate_embeddings_for_images(validation=True)


	def validation_step(self, batch, batch_idx):
		"""
		Validation step.

		Args:
			batch: Input batch for validation. Image, caption pairs.
			batch_idx: Index of the current batch. 
		"""

		mAP = self.shared_step(batch)
		self.log('validation mAP',np.mean(mAP), batch_size=self.batch_size)
		self.validation_step_outputs.append(mAP)

	def on_validation_epoch_end(self):
		"""
		Called at the end of the validation epoch to calculate and log the average mAP.
		"""
		avg_mAP = np.mean(np.concatenate(self.validation_step_outputs))
		self.log('avg_val_mAP', avg_mAP, batch_size=self.batch_size, prog_bar=True)

	def configure_optimizers(self):
		"""
		Configure the optimizer.

		Returns:
			toch.optim.Optimizer: Configured optimizer (AdamW)
		"""

		param_groups = define_param_groups(self, self.weight_decay, 'adam')

		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
		# lr_scheduler = LinearWarmupCosineAnnealingLR(
		# 	optimizer, warmup_epochs=10, max_epochs=self.max_epochs, warmup_start_lr=self.learning_rate/10
		# )
		return optimizer
