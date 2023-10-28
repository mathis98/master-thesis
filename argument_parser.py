import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(description="BERT Sentence Embedding")

	parser.add_argument("--rsicd", action="store_true", help="Use RSICD Dataset")
	parser.add_argument("--ucm", action="store_false", help="Use UCM dataset")
	parser.add_argument("--embedding", default="CLS", type=str, help="Text Embedding Method")

	return parser.parse_args()