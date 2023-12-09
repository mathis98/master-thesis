import lightning.pytorch as pl
from lightning.pytorch  import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from transformers import AutoTokenizer
from lightning.pytorch.accelerators import find_usable_cuda_devices

# Full pipeline
from model.full_pipeline import FullPipeline

from data.image.data_module import ImageDataModule
from data.text.data_module import SentenceDataModule
from data.imagetext.image_text_pair import ImageTextPairDataModule

from data.image.simclr_data_module import SimCLRDataModule as SimCLRImageDataModule
from data.text.simclr_data_module import SimCLRDataModule as SimCLRTextDataModule

# Argument parsing
from utility.argument_parser import parse_arguments


args = parse_arguments()

args.image_size = tuple(args.image_size)

seed_everything(42, workers=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# SimCLR
augmentation_transform = v2.Compose([
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

if args.intra == True:
	image_data_module = SimCLRImageDataModule(args.img_path, args.image_size, args.batch_size, augmentation_transform)
	image_data_module.prepare_data()
	image_data_module.setup(stage="fit")

	text_data_module = SimCLRTextDataModule(args.batch_size, args.text_path, tokenizer)
	text_data_module.prepare_data()
	text_data_module.setup()

elif args.intra == False:
	image_data_module = ImageDataModule(args.img_path, args.image_size, args.batch_size, args.num_repeats)
	image_data_module.prepare_data()
	image_data_module.setup(stage='fit')


	text_data_module = SentenceDataModule(args.model_name, args.batch_size, args.text_path)
	text_data_module.prepare_data()
	text_data_module.setup(stage='fit')


image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, args.batch_size)
image_text_pair_data_module.setup(stage='fit')

# RETURNS pairs from image_data_module, text_data_module so one of
# SIMCLR: ((original_image, augmented_image, image_path), (inputs, inputs_aug, sentence, aug_sentence, index))
# NO SIMCLR: ((image, image_path), (inputs, sentence, index))


full_pipeline = FullPipeline(
	batch_size=args.batch_size, 
	max_epochs=args.max_epochs, 
	temperature=args.temperature, 
	learning_rate=args.learning_rate, 
	weight_decay=args.weight_decay, 
	intra=args.intra,
	top_k=args.top_k,
	val_dataloader = image_text_pair_data_module.val_dataloader,
	test_dataloader = image_text_pair_data_module.test_dataloader,
)

logger = pl.loggers.CSVLogger('logs', name='full_pipeline_full_val_test')

logger.log_hyperparams(args)

# devices = find_usable_cuda_devices(1)
# print(f'training on GPU {devices}')

devices = [2]

print(f'training on GPU {devices}')

trainer = pl.Trainer(
	logger=logger, 
	accelerator='cuda', 
	devices=[1], 
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
		EarlyStopping(monitor='avg_val_mAP', min_delta=.0, patience=5, verbose=False, mode='max'),
		# StochasticWeightAveraging(swa_lrs=1e-2),
	],
	accumulate_grad_batches=args.accumulate,
)

trainer.fit(
	full_pipeline, 
	image_text_pair_data_module.train_dataloader(),
	image_text_pair_data_module.val_dataloader(),
)

trainer.test(ckpt_path='best', dataloaders=image_text_pair_data_module.test_dataloader())

