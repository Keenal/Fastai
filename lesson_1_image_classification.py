# -*- coding: utf-8 -*-

!pip install fastai --upgrade



from fastai import *
from fastai.vision import *

help(untar_data)

path = untar_data(URLs.PETS); path

path.ls()

path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)
fnames

np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.jpg$')

data = ImageDataBunch.from_name_re(path_img, fnames, path, ds_tfms=get_transforms(), size=224)
data = data.normalize(imagenet_stats)

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224)
data = data.normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(7,6))

data.show_batch(rows=3, figsize=(7,6))

print(data.classes)
len(data.classes),data.c

learn = create_cnn(data, models.resnet34, pretrained=True, metrics=error_rate)

learn = cnn_learner(data, models.resnet34, pretrained=True, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save('stage_1')

interp = ClassificaitonInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)

interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_top_losses(9, figsize=(15,11))

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=2)

learn.load('stage_1')

learn.unfreeze()

learn.fit_one_cycle(4)

learn.lr_find()

learn.recorder.plot()

learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(3e-05, 1e-05)) # 5% error rate

learn.fit_one_cycle(2, max_lr=slice(5e-05, 1e-05)) # 4% error rate

learn.fit_one_cycle(2, max_lr=slice(4.5e-05, 8.8e-04)) # 6% error rate

