import pytorch_lightning as pl

# Full pipeline
from model.full_pipeline import FullPipeline

from data.image.data_module import ImageDataModule
from data.text.data_module import SentenceDataModule
from data.imagetext.image_text_pair import ImageTextPairDataLoader

text_path = '../Datasets/UCM/dataset.json'
img_path = '../Datasets/UCM/imgs'
model_name = 'prajjwal1/bert-small'
image_size = (224, 224)
batch_size = 64
num_repeats = 5


image_data_module = ImageDataModule(img_path, image_size, batch_size, num_repeats)
image_data_module.prepare_data()
image_data_module.setup()


text_data_module = SentenceDataModule(model_name, batch_size, text_path)
text_data_module.prepare_data()
text_data_module.setup()

image_text_pair_dataloader = ImageTextPairDataLoader(image_data_module.train_dataloader(), text_data_module.train_dataloader())


full_pipeline = FullPipeline(batch_size)

trainer = pl.Trainer(fast_dev_run=True)

trainer.fit(
	full_pipeline, 
	image_text_pair_dataloader.train_dataloader()
)