import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import imageio


class SimCLRDataset(Dataset):
	def __init__(self, image_paths, image_size, transform):
		self.image_paths = image_paths
		self.image_size = image_size
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		image = Image.open(image_path).convert('RGB')
		original_image = self.transform(image)
		augmented_image = self.transform(image)

		return original_image, augmented_image


class SimCLRDataModule(pl.LightningDataModule):
	def __init__(self, data_dir, image_size, batch_size, augmentation_transform):
		super().__init__()
		self.data_dir = data_dir
		self.image_size = image_size
		self.batch_size = batch_size
		self.augmentation_transform = augmentation_transform

	def prepare_data(self):
		self.image_paths = [os.path.join(self.data_dir, filename) for filename in os.listdir(self.data_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]

	def setup(self, stage=None):
		self.train_dataset = SimCLRDataset(self.image_paths, self.image_size, self.augmentation_transform)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=87)