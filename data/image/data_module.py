import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import imageio
import numpy as np


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