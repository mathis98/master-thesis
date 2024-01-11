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
import yaml

batch_size = 512
num_repeats = 5

intra = False

augmentation_transform = v2.Compose([
		v2.Resize((224,224)),
		v2.RandAugment(), # “RandAugment: Practical automated data augmentation with a reduced search space”.
		v2.ToImageTensor(),
		v2.ConvertImageDtype(),
])

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

version = input('Version number to load: ')


if version == '':
	print('Loading untrained model')

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


	text_data_module = SentenceDataModule('prajjwal1/bert-small', batch_size, '../Datasets/UCM/dataset.json', 5)
	text_data_module.prepare_data()
	text_data_module.setup(stage='predict')

	hparams = {num_repeats:5}

	image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, batch_size)
	image_text_pair_data_module.setup(stage='predict')

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
		dataset='ucm',
		num_repeats=5,
	)

else:
	name = os.listdir(f'./logs/full_pipeline_full_val_test/version_{version}/checkpoints')[0]
	checkpoint = f'./logs/full_pipeline_full_val_test/version_{version}/checkpoints/{name}'

	print(f'Loading from {checkpoint}')

	with open(f'./logs/full_pipeline_full_val_test/version_{version}/hparams.yaml') as file:
		hparams = yaml.safe_load(file)

	if not 'technique' in hparams:
		hparams['technique'] = 'Repeat'

	if not 'dataset' in hparams:
		hparams['dataset'] = 'ucm'

	print(hparams)

	image_data_module = ImageDataModule(hparams['img_path'], tuple(hparams['image_size']), hparams['batch_size'], num_repeats=hparams['num_repeats'], technique=hparams['technique'])
	image_data_module.prepare_data()
	image_data_module.setup(stage='predict')

	text_data_module = SentenceDataModule(hparams['model_name'], hparams['batch_size'], hparams['text_path'], num_repeats=hparams['num_repeats'], technique=hparams['technique'])
	text_data_module.prepare_data()
	text_data_module.setup(stage='predict')

	image_text_pair_data_module = ImageTextPairDataModule(image_data_module, text_data_module, hparams['batch_size'])
	image_text_pair_data_module.setup(stage='predict')

	full_pipeline = FullPipeline.load_from_checkpoint(
		checkpoint,
		batch_size=hparams['batch_size'], 
		max_epochs=hparams['max_epochs'], 
		temperature=hparams['temperature'], 
		learning_rate=hparams['learning_rate'], 
		weight_decay=hparams['weight_decay'], 
		intra=hparams['intra'],
		top_k=hparams['top_k'],
		val_dataloader = image_text_pair_data_module.val_dataloader,
		test_dataloader = image_text_pair_data_module.test_dataloader,
		dataset=hparams['dataset'],
		num_repeats=hparams['num_repeats'],
	)

device = 'cuda:2'

full_pipeline.resnet_embedding_module.to(device)
full_pipeline.bert_embedding_module.to(device)
full_pipeline.projection_head.to(device)

image_text_pair_data_module.device = device


full_pipeline.eval()

image_embeddings, labels = full_pipeline.calculate_embeddings_for_images(validation=False, true_label=True)


# print('text:')
# print(list(text_data_module.test_dataset))
# print('image:')
# print(list(image_data_module.test_dataset))
# print('imagetext:')
# print(list(image_text_pair_data_module.test_dataset)[:5])

if hparams['dataset'] == 'nwpu':
	categories = sorted(os.listdir(hparams['img_path']))

random_sample = random.sample(list(text_data_module.test_dataset), 5)

print('5 Random samples:')
for element in random_sample:

	name = ''

	if hparams['dataset'] == 'nwpu':
		index = element[2] // hparams["num_repeats"] + 1
		category_index = index // 700
		elem_index = index % 700

		name = f' {categories[category_index]}_{elem_index}'

	print(f'Sentence: {element[1]} (Index: {element[2] // hparams["num_repeats"] + 1}{name})')

while True:

	query = input('Enter query caption (Ctrl + C to exit): ')
	if not query:
		break

	caption = tokenizer(query, return_tensors='pt').to(device)
	new_caption = [caption]
	new_caption_embedding = full_pipeline.bert_embedding_module(new_caption)
	new_caption_projection = full_pipeline.projection_head(new_caption_embedding)

	similarity_scores = torch.nn.functional.cosine_similarity(image_embeddings, new_caption_projection)

	top_k = 100
	sorted_indices = torch.argsort(similarity_scores, descending=True)[:top_k]

	idxs = []

	print('20 closest images:')
	for idx in sorted_indices:

		name = ''

		index = int(labels[idx].item()) + 1
		idxs.append(index)

		if hparams['dataset'] == 'nwpu':
			category_index = index // 700
			elem_index = index % 700

			name = f' {categories[category_index]}_{elem_index}'

		print(f'Image index: {index}{name}, Similarity: {similarity_scores[idx].item()}')

	output = {
		'dataset': hparams['dataset'],
		'query': query,
		'idxs': idxs
	}

	print(output)
