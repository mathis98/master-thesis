import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.transforms import v2
from transformers import AutoTokenizer
import torchvision
from lightning.pytorch.accelerators import find_usable_cuda_devices

# Full pipeline
from model.full_pipeline import FullPipeline

from data.image.data_module import ImageDataModule
from data.text.data_module import SentenceDataModule
from data.imagetext.image_text_pair import ImageTextPairDataModule

from data.image.simclr_data_module import SimCLRDataModule as SimCLRImageDataModule
from data.text.simclr_data_module import SimCLRDataModule as SimCLRTextDataModule

torchvision.disable_beta_transforms_warning()

text_path = '../Datasets/UCM/dataset.json'
img_path = '../Datasets/UCM/imgs'
model_name = 'prajjwal1/bert-small'
image_size = (224, 224)
batch_size = 64
num_repeats = 5
max_epochs = 500

intra = True

tokenizer = AutoTokenizer.from_pretrained(model_name)

# SimCLR
augmentation_transform = v2.Compose([
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

if intra == True:
	image_data_module = SimCLRImageDataModule(img_path, image_size, batch_size, augmentation_transform)
	image_data_module.prepare_data()
	image_data_module.setup(stage="fit")

	text_data_module = SimCLRTextDataModule(batch_size, text_path, tokenizer)
	text_data_module.prepare_data()
	text_data_module.setup()

elif intra == False:
	image_data_module = ImageDataModule(img_path, image_size, batch_size, num_repeats)
	image_data_module.prepare_data()
	image_data_module.setup(stage='fit')


	text_data_module = SentenceDataModule(model_name, batch_size, text_path)
	text_data_module.prepare_data()
	text_data_module.setup(stage='fit')


image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, batch_size)
image_text_pair_data_module.setup(stage='fit')

# RETURNS pairs from image_data_module, text_data_module so one of
# SIMCLR: ((original_image, augmented_image, image_path, source_image), (inputs, inputs_aug, sentence, aug_sentence, index))
# NO SIMCLR: ((image, image_path), (inputs, sentence, index))


full_pipeline = FullPipeline(batch_size, intra=intra)

logger = pl.loggers.CSVLogger('logs', name='full_pipeline_simple')

# devices = find_usable_cuda_devices(1)
# print(f'training on GPU {devices}')

trainer = pl.Trainer(
	logger=logger, 
	# accelerator='cpu', 
	# devices=devices, 
	max_epochs=max_epochs,
	callbacks=[
		ModelCheckpoint(
			save_weights_only=True, 
			mode='max', 
			monitor='avg_val_mAP', 
			filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
		),
		LearningRateMonitor('epoch'),
		# EarlyStopping(monitor='avg_val_mAP', min_delta=.0, patience=5, verbose=False, mode='max'),
	]
)

trainer.fit(
	full_pipeline, 
	image_text_pair_data_module.train_dataloader(),
	image_text_pair_data_module.val_dataloader(),
)

trainer.test(full_pipeline, dataloaders=image_text_pair_data_module.test_dataloader())

