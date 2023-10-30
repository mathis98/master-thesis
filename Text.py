from torchinfo import summary

from argument_parser import parse_arguments
from util import closest_indices

from data.text.data_module import SentenceDataModule

from model.text_embedding import BERTSentenceEmbedding

args = parse_arguments()




model_name = 'prajjwal1/bert-mini'
batch_size = 64

path = '../Datasets/UCM/dataset.json' if not args.ucm else '../Datasets/RSICD/dataset_rsicd.json'

print(f"Using {path}")
print(f"Using {args.embedding}")

data_module = SentenceDataModule(model_name, batch_size, path)
data_module.prepare_data()
data_module.setup()

bert_embedding = BERTSentenceEmbedding(model_name, args.embedding)

summary(bert_embedding)

trainer = pl.Trainer()
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