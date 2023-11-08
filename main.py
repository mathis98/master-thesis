import pytorch_lightning as pl
from torchvision.transforms import v2

# Full pipeline
from model.full_pipeline import FullPipeline

from data.image.data_module import ImageDataModule
from data.text.data_module import SentenceDataModule
from data.imagetext.image_text_pair import ImageTextPairDataModule

from data.image.simclr_data_module import SimCLRDataModule as SimCLRImageDataModule
from data.text.simclr_data_module import SimCLRDataModule as SimCLRTextDataModule

text_path = '../Datasets/UCM/dataset.json'
img_path = '../Datasets/UCM/imgs'
model_name = 'prajjwal1/bert-small'
image_size = (224, 224)
batch_size = 64
num_repeats = 5

simclr = False

# SimCLR
augmentation_transform = v2.Compose([
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

if simclr == True:
	image_data_module = SimCLRImageDataModule(img_path, image_size, batch_size, augmentation_transform)
	image_data_module.prepare_data()
	image_data_module.setup(stage="fit")

	text_data_module = SimCLRTextDataModule(model_name, batch_size, text_path)
	text_data_module.prepare_data()
	text_data_module.setup()

elif simclr == False:
	image_data_module = ImageDataModule(img_path, image_size, batch_size, num_repeats)
	image_data_module.prepare_data()
	image_data_module.setup(stage='fit')


	text_data_module = SentenceDataModule(model_name, batch_size, text_path)
	text_data_module.prepare_data()
	text_data_module.setup(stage='fit')


image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, batch_size)
image_text_pair_data_module.setup(stage='fit')

# print('image: ')
# print(image_data_module.train_dataset.image_paths[:10])
# print('text: ')
# print(text_data_module.train_dataset.sentences[:10])
# print('both: ')
# elem = image_text_pair_data_module.train_dataset
# print(elem[0][1][:10], elem[1][1][:10])

# print(list(image_text_pair_data_module.train_dataloader())[:10])


full_pipeline = FullPipeline(batch_size, simclr=simclr)

logger = pl.loggers.CSVLogger('logs', name='full_pipeline_simple')
trainer = pl.Trainer(logger=logger)

trainer.fit(
	full_pipeline, 
	image_text_pair_data_module.train_dataloader(),
	image_text_pair_data_module.val_dataloader(),
)

trainer.test(full_pipeline, dataloaders=image_text_pair_data_module.test_dataloader())

