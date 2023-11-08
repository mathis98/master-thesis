import torch
import pytorch_lightning as pl
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as ResNet_Weights

import sys
sys.path.append('..')

from loss.contrastive_loss import SimCLRLoss


class SimCLRModule(pl.LightningModule):
	def __init__(self, image_size=(224, 224), temperature=.07, learning_rate=1e-4):
		super(SimCLRModule, self).__init__()
		self.model = resnet(weights=ResNet_Weights.DEFAULT)
		self.temperature = temperature
		self.criterion = SimCLRLoss(temperature)
		self.learning_rate = learning_rate

	def forward(self, original, augmented):
		z_original = self.model(original)
		z_augmented = self.model(augmented)
		return z_original, z_augmented

	def source_image_embeddings(self, source_images):
		return self.model(source_images)

	def training_step(self, batch, batch_idx):
		original, augmented = batch
		print(original[0])
		z_original, z_augmented = self(original, augmented)
		loss = self.criterion(z_original, z_augmented)
		self.log("train_loss", loss)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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