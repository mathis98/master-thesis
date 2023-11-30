from lightning.pytorch.cli import LightningCLI

from model.full_pipeline import FullPipeline
from data.imagetext.image_text_pair import ImageTextPairDataModule

LightningCLI(FullPipeline, ImageTextPairDataModule)