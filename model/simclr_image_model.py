import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as ResNet_Weights

import sys
sys.path.append('..')

from loss.contrastive_loss import SimCLRLoss


class SimCLRModule(pl.LightningModule):
	"""
	SimCLR Module for Self-Supervised Contrastive Learning on Images.

	Args:
		image_size (tuple): Size of the input images (height, width).
		temperature (float): Temperature parameter for NT-Xent loss.
		learning_rate (float): Learning rate for AdamW optimizer.
		hidden_dim (int): Dimension of the hidden layer in the projection head.
		weight_decay (float): Weight decay of AdamW (L2 Regularization).
		max_epochs (int): Maximum number of training epochs.

	Attributes:
		model (torchvision.models.resnet.Resnet): Pretrained ResNet model.
		temperature (float): Temperature parameter for NT-Xent loss.
		criterion (loss.contrastive_loss.SimCRLLoss): Contrastive loss function (NT-Xent).
		learning_rate (float): Learning rate for AdamW optimizer.
		weight_decay (float): Weight decay for AdamW (L2 Regularization).
		max_epochs (int): Maximum number of training epochs.
		hidden_dim (int): Dimension of the hidden layer in the projection head.
		projection_head (torch.nn.Sequential): Projection head for embedding.

	Methods:
		forward(batch): Forward pass through the SimCLR image model.
		training_step(batch, batch_idx): Training step for the SimCLR image model.
		configure_optimizers(): Configure the AdamW optimizer and learning rate scheduler.
	"""

	def __init__(self, image_size=(224, 224), temperature=.07, learning_rate=1e-4, hidden_dim=128, weight_decay=1e-4, max_epochs=300):
		super(SimCLRModule, self).__init__()
		
		self.model = resnet(weights=None)
		self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

		self.temperature = temperature
		self.criterion = SimCLRLoss(temperature=temperature)
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.max_epochs = max_epochs
		self.hidden_dim = hidden_dim

		self.projection_head = nn.Sequential(
			nn.Linear(512, 4*hidden_dim),
			nn.ReLU(),
			nn.Linear(4*hidden_dim, hidden_dim)
		)


	def forward(self, batch):
		"""
		Forward pass through the SimCLR image model.

		Args:
			batch (tuple): Tuple containing original and augmented images.

		Returns:
			torch.Tensor: Projected embeddings for original and augmented images.
		"""

		original, augmented, _, _ = batch
		z_original = self.model(original)
		z_original = z_original.view(z_original.size(0), -1)
		z_original = self.projection_head(z_original)

		z_augmented = self.model(augmented)
		z_augmented = z_augmented.view(z_augmented.size(0), -1)
		z_augmented = self.projection_head(z_augmented)

		return z_original, z_augmented

	def training_step(self, batch, batch_idx):
		"""
		Training step for the SimCLR image model.

		Args:
			batch (tuple): Tuple containing original and augmented images.
			batch_idx (int): Batch index.

		Returns:
			torch.Tensor: NT-Xent loss between original and augmented image.
		"""
		
		z_original, z_augmented = self(batch)
		z_original = z_original.squeeze()
		z_augmented = z_augmented.squeeze()
		loss = self.criterion(z_original, z_augmented)
		self.log("train_loss", loss, prog_bar=True)
		return loss

	def configure_optimizers(self):
		"""
		Configure the optimizer and learning rate scheduler.

		Returns:
			list: List containing the optimizer.
			list: List containing the learning rate scheduler.
		"""

		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=self.max_epochs, eta_min=self.learning_rate / 50
		)
		return [optimizer], [lr_scheduler]