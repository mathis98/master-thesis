import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18 as resnet
from torchvision.models import ResNet18_Weights as ResNet_Weights

import sys
sys.path.append('..')

from loss.contrastive_loss import SimCLRLoss


class SimCLRModule(pl.LightningModule):
	def __init__(self, image_size=(224, 224), temperature=.07, learning_rate=1e-4):
		super(SimCLRModule, self).__init__()
		self.model = resnet(weights=ResNet_Weights.DEFAULT)
		self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
		self.temperature = temperature
		self.criterion = SimCLRLoss(temperature=temperature)
		self.learning_rate = learning_rate

		self.model.fc = nn.Sequential(self.model.fc, nn.ReLu, nn.Linear(512, hidden_dim))

	def forward(self, original, augmented):
		z_original = self.model(original)
		z_augmented = self.model(augmented)
		return z_original, z_augmented

	def training_step(self, batch, batch_idx):
		original, augmented, _, _ = batch
		z_original, z_augmented = self(original, augmented)
		z_original = z_original.squeeze()
		z_augmented = z_augmented.squeeze()
		loss = self.criterion(z_original, z_augmented)
		# self.log("train_loss", loss)
		print('loss: ', loss)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer

	def embed_data(self, dataloader):
		embeddings = []

		self.model.eval()

		with torch.no_grad():
			for batch in dataloader:
				original, _, _, _ = batch
				z_original, _ = self(original, original)
				embeddings.append(z_original)

		embeddings = torch.vstack(embeddings)
		return embeddings