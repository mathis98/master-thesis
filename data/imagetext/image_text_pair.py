import pytorch_lightning as pl

class ImageTextPairDataLoader(pl.LightningDataModule):
	def __init__(self, image_dataloader, text_dataloader):
		super(ImageTextPairDataLoader, self).__init__()
		self.image_dataloader = image_dataloader
		self.text_dataloader = text_dataloader

	def prepare_data(self):
		pass

	def setup(self, stage=None):
		pass

	def train_dataloader(self):
		for image_batch, text_batch in zip(
			self.image_dataloader.train_dataloader(), 
			self.text_dataloader.train_dataloader()
		):
			yield image_batch, text_batch

	def val_dataloader(self):
		for image_batch, text_batch in zip(
			self.image_dataloader.val_dataloader(), 
			self.text_dataloader.val_dataloader()
		):
			yield image_batch, text_batch


	def test_dataloader(self):
		for image_batch, text_batch in zip(
			self.image_dataloader.test_dataloader(), 
			self.text_dataloader.test_dataloader()
		):
			yield image_batch, text_batch