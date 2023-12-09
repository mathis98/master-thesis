import torch
import pytorch_lightning as pl
from torchvision.models import resnet34 as resnet
from torchvision.models import ResNet34_Weights as ResNet_Weights


class ImageEmbeddingModule(pl.LightningModule):
	"""
	Image Embedding model using a ResNet34-based model.

	Args:
		image_size (tuple): Tuple representing the input image size (height, width).
	"""

	def __init__(self, image_size=(224, 224)):
		super(ImageEmbeddingModule, self).__init__()

		self.model = resnet(weights=ResNet_Weights, num_classes=512)
		self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) # Remove classification layer

	def forward(self, batch):
		"""
		Forward pass through the image embedding module.

		Args: 
			batch (tuple): Input batch containing images.

		Returns:
			torch.Tensor: Image embeddings.
		"""

		return self.model(batch[0])