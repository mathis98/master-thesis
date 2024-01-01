import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset


class ImageTextPairDataset(Dataset):
	"""
	Dataset class for pairs of images and captions.

	Args:
		image_dataset: Dataset for images.
		text_dataset: Dataset for captions.
	"""

	def __init__(self, image_dataset, text_dataset):
		self.image_dataset = image_dataset
		self.text_dataset = text_dataset

	def __len__(self):
		return len(self.image_dataset)

	def __getitem__(self, idx):
		return self.image_dataset[idx], self.text_dataset[idx]


class ImageTextPairDataModule(pl.LightningDataModule):
	"""
	A Data Module for pairs of image and text

	Args:
		image_data_module: data module for handling images
		text_data_module: data module for handling captions
		batch_size: batch size returned by dataloader
	"""
	def __init__(self, image_data_module, text_data_module, batch_size=64):
		super(ImageTextPairDataModule, self).__init__()
		self.image_data_module = image_data_module
		self.text_data_module = text_data_module
		self.batch_size = batch_size

	def prepare_data(self):
		"""
		Prepare data for both image and caption data modules
		"""

		self.image_data_module.prepare_data()
		self.text_data_module.prepare_data()

	def setup(self, stage=None):
		"""
		Sets up dataset for training, validation, and testing.

		Args:
			stage: Stage of training (None for overall setup).
		"""

		self.image_data_module.setup(stage)
		self.text_data_module.setup(stage)

		self.dataset = ImageTextPairDataset(self.image_data_module.dataset, self.text_data_module.dataset)

		self.train_dataset = ImageTextPairDataset(self.image_data_module.train_dataset, self.text_data_module.train_dataset)
		self.val_dataset = ImageTextPairDataset(self.image_data_module.val_dataset, self.text_data_module.val_dataset)
		self.test_dataset = ImageTextPairDataset(self.image_data_module.test_dataset, self.text_data_module.test_dataset)

	def dataloader(self):
		return DataLoader(self.dataset, self.batch_size, num_workers=30, pin_memory=True)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, self.batch_size, num_workers=30, shuffle=True, drop_last=True, pin_memory=True)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, self.batch_size, num_workers=30, pin_memory=True)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, self.batch_size, num_workers=30, pin_memory=True)