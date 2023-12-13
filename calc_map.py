import lightning.pytorch as pl
import torch
from model.full_pipeline import FullPipeline
from data.image.data_module import ImageDataModule
from data.text.data_module import SentenceDataModule
from data.imagetext.image_text_pair import ImageTextPairDataModule
from data.image.simclr_data_module import SimCLRDataModule as SimCLRImageDataModule
from data.text.simclr_data_module import SimCLRDataModule as SimCLRTextDataModule
from torchvision.transforms import v2
from transformers import AutoTokenizer
import random
import os

batch_size = 512

intra = False

augmentation_transform = v2.Compose([
		v2.Resize((224,224)),
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')


if intra == True:
	image_data_module = SimCLRImageDataModule('../Datasets/UCM/imgs', (224,224), batch_size, augmentation_transform)
	image_data_module.prepare_data()
	image_data_module.setup(stage="predict")

	text_data_module = SimCLRTextDataModule(batch_size, '../Datasets/UCM/dataset.json', tokenizer)
	text_data_module.prepare_data()
	text_data_module.setup()

elif intra == False:
	image_data_module = ImageDataModule('../Datasets/UCM/imgs', (224,224), batch_size, 5)
	image_data_module.prepare_data()
	image_data_module.setup(stage='predict')


	text_data_module = SentenceDataModule('prajjwal1/bert-small', batch_size, '../Datasets/UCM/dataset.json')
	text_data_module.prepare_data()
	text_data_module.setup(stage='predict')

image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, batch_size)
image_text_pair_data_module.setup(stage='predict')

version = input('Version number to load: ')


if version == '':
	print('Loading untrained model')

	full_pipeline = FullPipeline(
		batch_size=batch_size, 
		max_epochs=1, 
		temperature=3.0, 
		learning_rate=1e-4, 
		weight_decay=1e-4, 
		intra=intra,
		top_k=20,
		val_dataloader = image_text_pair_data_module.val_dataloader,
		test_dataloader = image_text_pair_data_module.test_dataloader,
	)

else:
	name = os.listdir(f'./logs/full_pipeline_full_val_test/version_{version}/checkpoints')[0]
	checkpoint = f'./logs/full_pipeline_full_val_test/version_{version}/checkpoints/{name}'

	print(f'Loading from {checkpoint}')

	full_pipeline = FullPipeline.load_from_checkpoint(
		checkpoint,
		batch_size=batch_size, 
		max_epochs=1, 
		temperature=3.0, 
		learning_rate=1e-4, 
		weight_decay=1e-4, 
		intra=intra,
		top_k=20,
		val_dataloader = image_text_pair_data_module.val_dataloader,
		test_dataloader = image_text_pair_data_module.test_dataloader,
	)

device = 'cuda:3'

full_pipeline.resnet_embedding_module.to(device)
full_pipeline.bert_embedding_module.to(device)
full_pipeline.projection_head.to(device)

image_text_pair_data_module.device = device


logger = pl.loggers.CSVLogger('logs', name='full_pipeline_full_val_test')

logger.log_hyperparams(args)

trainer = pl.Trainer(
	logger=logger, 
	accelerator='cuda', 
	devices=[2], 
	max_epochs=args.max_epochs,
	log_every_n_steps=5,
	gradient_clip_val=0.5,
	precision='16-mixed',
	callbacks=[
		ModelCheckpoint(
			save_weights_only=True, 
			mode='max', 
			monitor='avg_val_mAP', 
			filename='{epoch}-{avg_val_mAP:.2f}-{validation mAP:.2f}'
		),
		LearningRateMonitor('epoch'),
		EarlyStopping(monitor='avg_val_mAP', min_delta=.0, patience=10, verbose=False, mode='max'),
		# StochasticWeightAveraging(swa_lrs=1e-2),
	],
	accumulate_grad_batches=args.accumulate,
)

trainer.test(ckpt_path='best', dataloaders=image_text_pair_data_module.test_dataloader())