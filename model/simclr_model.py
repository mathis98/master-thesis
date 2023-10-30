import torch
import pytorch_lightning as pl
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as ResNet_Weights

import sys
sys.path.append('..')

from loss.contrastive_loss import SimCLRLoss


class SimCLRModule(pl.LightningModule):
	def __init__(self, image_size, temperature=.07):
		super(SimCLRModule, self).__init__()
		self.model = resnet(weights=ResNet_Weights.DEFAULT)
		self.temperature = temperature
		self.criterion = SimCLRLoss(temperature)

	def forward(self, original, augmented):
		z_original = self.model(original)
		z_augmented = self.model(augmented)
		return z_original, z_augmented

	def training_step(self, batch, batch_idx):
		original, augmented = batch
		z_original, z_augmented = self(original, augmented)
		loss = self.criterion(z_original, z_augmented)
		self.log("train_loss", loss)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=.001)
		return optimizer

	def embed_data(self, dataloader):
		embeddings = []

		self.model.eval()

		with torch.no_grad():
			for batch in dataloader:
				original, _ = batch
				z_original, _ = self(original, original)
				embeddings.append(z_original)

		embeddings = torch.vstack(embeddings)
		return embeddings