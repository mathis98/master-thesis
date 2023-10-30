import torch
import pytorch_lightning as pl
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as ResNet_Weights


class ImageEmbeddingModule(pl.LightningModule):
	def __init__(self, image_size):
		super(ImageEmbeddingModule, self).__init__()
		self.model = resnet(weights=ResNet_Weights.DEFAULT)
		self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) # Remove classification layer

	def forward(self, input):
		return self.model(input[0])