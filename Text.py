from torchinfo import summary
import pytorch_lightning as pl
import torch
import numpy as np
from argument_parser import parse_arguments

# Helper functions
from utility.helpers import closest_indices, visualize_text_augmentations

# Data Modules
from data.text.data_module import SentenceDataModule
from data.text.simclr_data_module import SimCLRDataModule

# Embedding Modules
from model.text_embedding import BERTSentenceEmbedding
from model.simclr_text_model import SimCLRModule

args = parse_arguments()


model_name = 'prajjwal1/bert-mini'
batch_size = 64
path = '../Datasets/UCM/dataset.json' if not args.ucm else '../Datasets/RSICD/dataset_rsicd.json'
simclr = False

print(f"Using {path}")
print(f"Using {args.embedding}")

# Embeddding only
data_module = SentenceDataModule(model_name, batch_size, path)
data_module.prepare_data()
data_module.setup()

bert_embedding = BERTSentenceEmbedding(model_name, args.embedding)


# SimCLR only
simclr_data_module = SimCLRDataModule(model_name, batch_size, path)
simclr_data_module.prepare_data()
simclr_data_module.setup()

simclr_module = SimCLRModule(model_name, args.embedding)


trainer = pl.Trainer(fast_dev_run=True)


if simclr:
	
	trainer.fit(simclr_module, simclr_data_module.train_dataloader())

	print('fitting is done!')

	visualize_text_augmentations(simclr_data_module.dataset, 5)

	embeddings = simclr_module.embed_data(simclr_data_module.train_dataloader())

else:

	summary(bert_embedding)

	embeddings = []

	with torch.no_grad():
		predictions = trainer.predict(bert_embedding, data_module.train_dataloader())


	for batch in predictions:
		embeddings.extend(batch.tolist())

	embeddings = np.vstack(embeddings)


print(f"Shape: {embeddings.shape}")

pairs = closest_indices(embeddings)

for first, second in pairs:
	print(data_module.dataset.sentences[first], '<-->', data_module.dataset.sentences[second], ' (', first, ',', second, ')')