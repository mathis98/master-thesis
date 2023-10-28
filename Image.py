import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from PIL import Image
import imageio
import numpy as np
from torchinfo import summary

from util import closest_indices

class ImageDataSet(Dataset):
	def __init__(self, image_paths, image_size):
		self.image_paths = image_paths
		self.image_size = image_size
		self.transform = transforms.Compose([
			transforms.Resize(self.image_size),
			transforms.ToTensor(),
        ])

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		image = Image.open(image_path).convert('RGB')
		image = self.transform(image)
		return image, image_path

class ImageDataModule(pl.LightningDataModule):
	def __init__(self, data_dir, image_size, batch_size):
		super().__init__()
		self.data_dir = data_dir
		self.image_size = image_size
		self.batch_size = batch_size

	def prepare_data(self):
		self.image_paths = [os.path.join(self.data_dir, filename) for filename in os.listdir(self.data_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]

	def setup(self, stage=None):
		self.train_dataset = ImageDataSet(self.image_paths, self.image_size)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)


class ImageEmbeddingModule(pl.LightningModule):
	def __init__(self, image_size):
		super(ImageEmbeddingModule, self).__init__()
		self.model = resnet18(pretrained=True)
		self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) # Remove classification layer

	def forward(self, input):
		return self.model(input[0])


data_dir = '../Datasets/UCM/imgs'
batch_size = 64
image_size = (224, 224)

data_module = ImageDataModule(data_dir, image_size, batch_size)
data_module.prepare_data()
data_module.setup()

image_embedding_model = ImageEmbeddingModule(image_size)

summary(image_embedding_model)

trainer = pl.Trainer()

with torch.no_grad():
	predictions = trainer.predict(image_embedding_model, dataloaders=data_module.train_dataloader())

embeddings = torch.vstack(predictions)
embeddings = embeddings.view(embeddings.size(0), -1)

print('Shape: ', embeddings.shape)

pairs = closest_indices(embeddings)

for first, second in pairs:
	print(data_module.train_dataset.image_paths[first], '<-->', data_module.train_dataset.image_paths[second], ' (', first, ',', second, ')')