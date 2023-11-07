import pytorch_lightning as pl
from torch.utils.data import DataLoader

class ImageTextPairDataModule(pl.LightningDataModule):
	def __init__(self, image_data_module, text_data_module):
		super(ImageTextPairDataModule, self).__init__()
		self.image_data_module = image_data_module
		self.text_data_module = text_data_module

	def prepare_data(self):
		pass

	def setup(self, stage=None):
		pass

	def train_dataloader(self):
		image_loader = self.image_data_module.train_dataloader()
		text_loader = self.text_data_module.train_dataloader()
		
		return zip(image_loader, text_loader)

	def val_dataloader(self):
		image_loader = self.image_data_module.val_dataloader()
		text_loader = self.text_data_module.val_dataloader()

		return zip(image_loader, text_loader)

	def test_dataloader(self):
		image_loader = self.image_data_module.test_dataloader()
		text_loader = self.text_data_module.test_dataloader()
		
		return zip(image_loader, text_loader)