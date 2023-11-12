import torch
import pytorch_lightning as pl
from data.image.data_module import ImageDataModule

data_dir = '../Datasets/UCM/imgs'
batch_size = 64
image_size = (224, 224)

data_module = ImageDataModule(data_dir, image_size, batch_size)
data_module.prepare_data()
data_module.setup()

loader = data_module.dataloader()


def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std
  
mean, std = batch_mean_and_sd(loader)
print("mean and std: \n", mean, std)