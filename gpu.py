import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

import fastai
print(fastai.__version__)

from fastai import *
from fastai.vision import *

path = untar_data(URLs.DOGS)

data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(1)

learn.unfreeze()
learn.fit_one_cycle(6, slice(1e-5,3e-4), pct_start=0.05)

accuracy(*learn.TTA())
