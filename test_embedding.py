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

batch_size = 512

intra = True

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


full_pipeline = FullPipeline.load_from_checkpoint(
	'./logs/full_pipeline_full_val_test/version_210/checkpoints/epoch=197-avg_val_mAP=0.88-validation mAP=0.92.ckpt',
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

device = 'cuda:0'

full_pipeline.resnet_embedding_module.to(device)
full_pipeline.bert_embedding_module.to(device)
full_pipeline.projection_head.to(device)

image_text_pair_data_module.device = device


full_pipeline.eval()

image_embeddings, labels = full_pipeline.calculate_embeddings_for_images(validation=False)

while True:

	new_caption = input('Enter query caption (Ctrl + C to exit): ')
	if not new_caption:
		break

	caption = tokenizer(new_caption, return_tensors='pt')
	new_caption[0] = new_caption
	new_caption_embedding = full_pipeline.bert_embedding_module(new_caption)
	new_caption_projection = full_pipeline.projection_head(new_caption_embedding)

	similarity_scores = torch.nn.function.cosine_similarity(new_caption_projection, image_embeddings)

	top_k = 5
	sorted_indices = torch.argsort(similarity_scores, descending=True)[:top_k]

	print('5 closest images:')
	for idx in sorted_indices:
		print(f'Image index: {labels[idx].item()}, Similarity: {similarity_scores[idx].item()}')
