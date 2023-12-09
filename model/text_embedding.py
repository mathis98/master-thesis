import torch
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer


class BERTSentenceEmbedding(pl.LightningModule):
	"""
	BERT-based sentence Embedding module.

	Args:
		model_name (str): Pretrained BERT model name.
		embedding (str): Type of sentence embedding to use ('pooler', 'CLS', 'last', last_n', 'sbert')

	Attributes:
		model_name (str): Pretrained BERT model name.
		tokenizer (transformers.AutoTokenizer): Tokenizer for the specified BERT model. 
		embedding (str): Type of sentence embedding.
		model (transforms.Automodel or sentence_transformers.SentenceTransformer): BERT or Sentence Transformer model. 

	Methods:
		forward(inputs): Forward pass through the model to obtain sentence embeddings. 
	"""

	def __init__(self, model_name='prajjwal1/bert-small', embedding='pooler'):
		super(BERTSentenceEmbedding, self).__init__()
		self.model_name = model_name
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.embedding = embedding

		if self.embedding == 'sbert':
			self.model = SentenceTransformer('all-mpnet-base-v2')

		else:
			self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

	def forward(self, inputs):
		"""
		Forward pass through the model to obtain sentence embeddings.

		Args:
			inputs (tuple): Tuple containing input tensors (input_ids, attention_mask).

		Returns:
			torch.Tensor: Computed sentence embeddings.
		"""

		print(inputs[0], self.embedding)

		if self.embedding != 'sbert':
			outputs = self.model(inputs[0]['input_ids'], inputs[0]['attention_mask'])

		if self.embedding == 'CLS':
			return outputs.last_hidden_state[:, 0, :]

		elif self.embedding == 'last':
			last_hidden = outputs.last_hidden_state
			return torch.mean(last_hidden, dim=1) 

		elif self.embedding == 'last_n':
			last_n_hidden = outputs.hidden_states[-4:]
			hiddens = torch.stack(last_n_hidden)
			resulting_states = torch.sum(hiddens, dim=0)

			return torch.mean(resulting_states, dim=1)

		elif self.embedding == 'pooler':
			return self.model.pooler(outputs.last_hidden_state)

		elif self.embedding == 'sbert':
			return self.model.encode(inputs[1])
