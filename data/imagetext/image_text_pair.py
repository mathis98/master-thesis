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
		for image_batch, text_batch in zip(self.image_dataloader, self.text_dataloader):
			yield image_batch, text_batch