import pytorch_lightning as pl
from torch.utils.data import DataLoader

class ImageTextPairDataModule(pl.LightningDataModule):
	def __init__(self, image_data_module, text_data_module, trainer, batch_size=64):
		super(ImageTextPairDataModule, self).__init__()
		self.image_data_module = image_data_module
		self.text_data_module = text_data_module
		self.batch_size = batch_size
		self.trainer = trainer

	def prepare_data(self):
		pass

	def setup(self, stage=None):
		pass

	def custom_collate_fn(self, batch):
		image_batch, text_batch = zip(*batch)
		return (image_batch, text_batch)

	def train_dataloader(self):
		image_loader = self.image_data_module.train_dataloader()
		text_loader = self.text_data_module.train_dataloader()
		loader = DataLoader(
			list(zip(image_loader, text_loader)),
			batch_size = self.batch_size,
			collate_fn = self.cusom_collate_fn,
		)
		return loader

	def val_dataloader(self):
		image_loader = self.image_data_module.val_dataloader()
		text_loader = self.text_data_module.val_dataloader()
		loader = DataLoader(
			list(zip(image_loader, text_loader)),
			batch_size = self.batch_size,
			collate_fn = self.cusom_collate_fn,
		)
		return loader


	def test_dataloader(self):
		image_loader = self.image_data_module.test_dataloader()
		text_loader = self.text_data_module.test_dataloader()
		loader = DataLoader(
			list(zip(image_loader, text_loader)),
			batch_size = self.batch_size,
			collate_fn = self.cusom_collate_fn,
		)
		return loader