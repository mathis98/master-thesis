import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import imageio
import numpy as np
from torchvision.transforms import v2
import torchvision

torchvision.disable_beta_transforms_warning()

basic_transform = v2.Compose([
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])


class SimCLRDataset(Dataset):
	"""
	Dataset class for SimCLR self-supervised contrastive learning on images.

	Args:
		image_paths: List of paths to images.
		image_size: Desired size of images.
		transform: Augmentation transform for images.
	"""

	def __init__(self, image_paths, image_size, transform):
		self.image_paths = image_paths
		self.image_size = image_size
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		image = Image.open(image_path).convert('RGB')
		original_image = basic_transform(image)
		augmented_image = self.transform(image)

		return original_image, augmented_image, image_path


class SimCLRDataModule(pl.LightningDataModule):
	"""
	Data module for SimCLR self-supervised contrastive learning on images.

	Args:
		data_dir: Directory containing image data.
		image_size: Desired size of the images.
		batch_size: Batch size for DataLoader.
		augmentation_transform: Augmentation transform for images.
		num_repeats: Number of times to repeat each image.
		seed: Seed for reproducibility.
	"""

	def __init__(self, data_dir, image_size, batch_size, augmentation_transform, num_repeats=5, seed=42):
		super().__init__()
		self.data_dir = data_dir
		self.image_size = image_size
		self.batch_size = batch_size
		self.augmentation_transform = augmentation_transform
		self.num_repeats = num_repeats
		self.seed = seed

	def prepare_data(self):
		"""
		Prepares image paths by repeating (for multiple captions) and shuffling.
		"""

		image_paths = [os.path.join(self.data_dir, filename) for filename in os.listdir(self.data_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
		image_paths = sorted(image_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))
		self.image_paths = np.repeat(image_paths, self.num_repeats)

	def setup(self, stage=None):
		"""
		Sets up datasets for training, validation, and testing.

		Args:
			stage: Stage of training (None for overall setup).
		"""

		total_size = len(self.image_paths)
		train_size = int(.8 * total_size)
		val_size = int(.1 * total_size)
		test_size = total_size - train_size - val_size

		indices = list(range(total_size))

		np.random.seed(self.seed)
		shuffled_indices = np.random.permutation(indices)
		
		train_indices = []
		val_indices = []
		test_indices = []

		elements_per_group = 100

		# Iterate through each group
		for group_start in range(0, len(indices), elements_per_group):
			group_end = group_start + elements_per_group
			group = indices[group_start:group_end]

			# Calculate the indices for train, val, and test
			train_end = int(len(group) * 0.8)
			val_end = train_end + int(len(group) * 0.1)

			# Split the group into train, val, and test
			train_indices.extend(group[:train_end])
			val_indices.extend(group[train_end:val_end])
			test_indices.extend(group[val_end:])

		self.dataset = SimCLRDataset([self.image_paths[i] for i in list(shuffled_indices)], self.image_size, self.augmentation_transform)

		self.train_dataset = SimCLRDataset([self.image_paths[i] for i in train_indices], self.image_size, self.augmentation_transform)
		self.val_dataset = SimCLRDataset([self.image_paths[i] for i in val_indices], self.image_size, self.augmentation_transform)
		self.test_dataset = SimCLRDataset([self.image_paths[i] for i in test_indices], self.image_size, self.augmentation_transform)

	def train_dataloader(self):
		"""
		Returns DataLoader for training data.

		Returns:
			DataLoader: DataLoader for training data.
		"""
		return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=30)