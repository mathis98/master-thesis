import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from torchinfo import summary
import pytorch_lightning as pl
import torch
import numpy as np
from lightning.pytorch.accelerators import find_usable_cuda_devices
from utility.argument_parser import parse_arguments
from transformers import AutoTokenizer

# Helper functions
from utility.helpers import closest_indices, visualize_text_augmentations

# Data Modules
from data.text.data_module import SentenceDataModule
from data.text.simclr_data_module import SimCLRDataModule

# Embedding Modules
from model.text_embedding import BERTSentenceEmbedding
from model.simclr_text_model import SimCLRModule

args = parse_arguments()


model_name = 'prajjwal1/bert-small'
batch_size = 256
path = '../Datasets/UCM/dataset.json'
max_epochs = 500
simclr = True

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Using {path}")
print(f"Using {args.embedding}")

# Embeddding only
data_module = SentenceDataModule(model_name, batch_size, path)
data_module.prepare_data()
data_module.setup()

bert_embedding = BERTSentenceEmbedding(model_name, args.embedding)


# SimCLR only
simclr_data_module = SimCLRDataModule(batch_size, path, tokenizer)
simclr_data_module.prepare_data()
simclr_data_module.setup()

simclr_module = SimCLRModule(model_name, args.embedding)


devices = find_usable_cuda_devices(1)
print(f'training on GPU {devices}')

trainer = pl.Trainer(accelerator='cuda', devices=devices, max_epochs=max_epochs)

# trainer = pl.Trainer()


if simclr:

	summary(simclr_module)
	visualize_text_augmentations(simclr_data_module.train_dataset, 5)
	
	trainer.fit(simclr_module, simclr_data_module.train_dataloader())

	with torch.no_grad():
		predictions = trainer.predict(simclr_module, simclr_data_module.train_dataloader())
		predictions = [elem[0] for elem in predictions]

		print(len(predictions))

else:

	summary(bert_embedding)

	with torch.no_grad():
		predictions = trainer.predict(bert_embedding, data_module.train_dataloader())

embeddings = torch.vstack(predictions)
embeddings = embeddings.view(embeddings.size(0), -1)


print(f"Shape: {embeddings.shape}")

pairs = closest_indices(embeddings)

if simclr:
	data_module = simclr_data_module

for first, second in pairs:
	print(data_module.train_dataset.sentences[first], '<-->', data_module.train_dataset.sentences[second], ' (', data_module.train_dataset.indices[first], ',', data_module.train_dataset.indices[second], ')')