import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import imageio
import numpy as np

torchvision.disable_beta_transforms_warning()


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
	def __init__(self, data_dir, image_size, batch_size, num_repeats=5, seed=42):
		super().__init__()
		self.data_dir = data_dir
		self.image_size = image_size
		self.batch_size = batch_size
		self.num_repeats = num_repeats
		self.seed = seed

	def prepare_data(self):
		image_paths = [os.path.join(self.data_dir, filename) for filename in os.listdir(self.data_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
		image_paths = sorted(image_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))
		self.image_paths = np.repeat(image_paths, self.num_repeats)

	def setup(self, stage=None):
		total_size = len(self.image_paths)
		train_size = int(.8 * total_size)
		val_size = int(.1 * total_size)
		test_size = total_size - train_size - val_size

		indices = list(range(total_size))

		np.random.seed(self.seed)
		shuffled_indices = np.random.permutation(indices)
		
		train_indices, val_indices, test_indices = shuffled_indices[:train_size], shuffled_indices[train_size:(train_size+val_size)], shuffled_indices[(train_size+val_size):]

		# print('image paths:')
		# print(self.image_paths[0:10])

		self.dataset = ImageDataSet(self.image_paths, self.image_size)

		self.train_dataset = ImageDataSet([self.image_paths[i] for i in train_indices], self.image_size)
		self.val_dataset = ImageDataSet([self.image_paths[i] for i in val_indices], self.image_size)
		self.test_dataset = ImageDataSet([self.image_paths[i] for i in test_indices], self.image_size)

	def dataloader(self):
		return DataLoader(self.dataset, batch_size=self.batch_size)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=30)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=30)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=30)
