import configargparse

def parse_arguments():
	"""
	Parse arguments from config file or CLI.

	Returns:
		configargparse.Namespace: Parsed arguments.
	"""

	p = configargparse.ArgParser(description='Full Pipeline: Image-Caption -> Projection')

	defaults = {
		'config': './config/default.cfg',
		'batch_size': 512,
		'intra': False,
		'temperature': 0.6,
		'learning_rate': 5e-4,
		'weight_decay': 1e-4,
		'hidden_dim': 128,
		'text_path': '../Datasets/UCM/dataset.json',
		'img_path': '../Datasets/UCM/imgs',
		'model_name': 'prajjwal1/bert-small',
		'image_size': [224,224],
		'num_repeats': 5,
		'max_epochs': 100,
		'embedding': 'pooler',
		'accumulate': 1,
		'top_k': 10,
		'technique': 'Concat',
	}

	p.add('--config', required=False, default=defaults['config'], is_config_file=True, help=f'config file path (default: {defaults["config"]})')
	p.add('--batch_size', type=int, default=defaults['batch_size'], help=f'Batch size for training (default: {defaults["batch_size"]})')
	p.add('--intra', action='store_true', help=f'Use intra-modal training (default: {defaults["intra"]})')
	p.add('--temperature', type=float, default=defaults['temperature'], help=f'Temperature for SimCLR loss (default: {defaults["temperature"]})')
	p.add('--learning_rate', type=float, default=defaults['learning_rate'], help=f'Learning rate for optimizer (default: {defaults["learning_rate"]})')
	p.add('--weight_decay', type=float, default=defaults['weight_decay'], help=f'Weight decay for optimizer (default: {defaults["weight_decay"]})')
	p.add('--hidden_dim', type=int, default=defaults['hidden_dim'], help=f'Hidden dimension for the projection head (default: {defaults["hidden_dim"]})')
	p.add('--text_path', type=str, default=defaults['text_path'], help=f'Path to the text dataset (default: {defaults["text_path"]})')
	p.add('--img_path', type=str, default=defaults['img_path'], help=f'Path to the image dataset (default: {defaults["img_path"]})')
	p.add('--model_name', type=str, default=defaults['model_name'], help=f'Pretrained model name (default: {defaults["model_name"]})')
	p.add('--image_size', nargs=2, type=int, default=defaults['image_size'], help=f'Image size (height and width) (default: {defaults["image_size"]})')
	p.add('--num_repeats', type=int, default=defaults['num_repeats'], help=f'Number of repeats for non-SimCLR data module (default: {defaults["num_repeats"]})')
	p.add('--max_epochs', type=int, default=defaults['max_epochs'], help=f'Maximum number of training epochs (default: {defaults["max_epochs"]})')
	p.add('--embedding', type=str, default=defaults['embedding'], help=f'Embedding strategy for BERT (default: {defaults["embedding"]})')
	p.add('--accumulate', type=int, default=defaults['accumulate'], help=f'Accumulate gradient batches (default: {defaults["accumulate"]})')
	p.add('--top_k', type=int, default=defaults['top_k'], help=f'Value for map@k (default: {defaults["top_k"]})')
	p.add('--technique', type=str, default=defaults['technique'], help=f'Technique for multiple queries (default: {defaults["technique"]})')

	options =  p.parse_args()
	print(p.format_values())

	return options
