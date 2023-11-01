import pytorch_lightning as pl

from model.full_pipeline import FullPipeline

path = '../Datasets/UCM/dataset.json'
data_dir = '../Datasets/UCM/imgs'
batch_size = 64

full_pipeline = FullPipeline(path, data_dir, batch_size)

trainer = pl.Trainer(fast_dev_run=True)

trainer.fit(full_pipeline)