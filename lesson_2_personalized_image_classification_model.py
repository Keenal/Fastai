
from google.colab import drive 
drive.mount('/content/gdrive') 

from fastai import *
from fastai.vision import *

folder = 'kathak'
file = 'kathak.csv'

path = Path('gdrive/My Drive/FastaiData')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

download_images(path/file, dest, max_pics=200)

folder = 'bharatnatyam'
file = 'bharatnatyam.csv'

path = Path('gdrive/My Drive/FastaiData')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

download_images(path/file, dest, max_pics=200)

classes = ['kathak', 'bharatnatyam']

for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

data

data.classes

data.show_batch(rows=3, figsize=(7,8))

data.classes, data.c, len(data.train_ds), len(data.valid_ds)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4) #18% error rate

learn.fit_one_cycle(4) #10% error rate

learn.fit_one_cycle(4) #7% error rate

learn.save('stage-1')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-3)) # 5% errror rate

learn.save('stage-2')

learn.load('stage-2');

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()

