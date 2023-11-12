import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as ResNet_Weights

import sys
sys.path.append('..')

from loss.contrastive_loss import SimCLRLoss


class SimCLRModule(pl.LightningModule):
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
		original, augmented, _, _ = batch
		z_original = self.model(original)
		z_original = z_original.view(z_original.size(0), -1)
		z_original = self.projection_head(z_original)

		z_augmented = self.model(augmented)
		z_augmented = z_augmented.view(z_augmented.size(0), -1)
		z_augmented = self.projection_head(z_augmented)

		return z_original, z_augmented

	def training_step(self, batch, batch_idx):
		
		z_original, z_augmented = self(batch)
		z_original = z_original.squeeze()
		z_augmented = z_augmented.squeeze()
		loss = self.criterion(z_original, z_augmented)
		self.log("train_loss", loss, prog_bar=True)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=self.max_epochs, eta_min=self.learning_rate / 50
		)
		return [optimizer], [lr_scheduler]