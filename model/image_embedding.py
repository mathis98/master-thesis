import torch
import pytorch_lightning as pl
from torchvision.models import resnet34 as resnet
from torchvision.models import ResNet34_Weights as ResNet_Weights


class ImageEmbeddingModule(pl.LightningModule):
	def __init__(self, image_size=(224, 224)):
		super(ImageEmbeddingModule, self).__init__()

		self.model = resnet(weights=ResNet_Weights)
		# self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) # Remove classification layer

	def forward(self, batch):
		return self.model(batch[0])