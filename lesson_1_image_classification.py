!pip install fastai --upgrade

from fastai import * # need to import libraries as deep learning uses a lot of libraries to get the task done so this way we can easily get our hands on multiple libraries at once
from fastai.vision import *

help(untar_data) # place the url as the string, file name is either the path or a string and destination is again path or string

path = untar_data(URLs.PETS); path #this will download our dataset, untar downloads it automatically

path.ls()

path_anno = path/'annotations' #better to use path objects than strings
path_img = path/'images'

fnames = get_image_files(path_img)
fnames

np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.jpg$') #reg expression that would extract the label names from the file names

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224) #tfms - center crops the image and resizing and padding
data = data.normalize(imagenet_stats) #make all our images of the same size so it normalizes our databunch objs

data.show_batch(rows=3, figsize=(7,6)) #look at your data to make sure it's good

print(data.classes)
len(data.classes),data.c #also look at the labels to confirm that are as many labels you wanted

learn = create_cnn(data, models.resnet34, pretrained=True, metrics=error_rate) # deprecated, use cnn_learner instead (next line)

learn = cnn_learner(data, models.resnet34, pretrained=True, metrics=error_rate) #this is how we train the data and it downloads the resnet34 pretrained weights

learn.fit_one_cycle(4) #the number of epocs you want it to run for

learn.save('stage_1')

interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses() #loss funcs tell you how good was your prediction, things that we were most confident about that we got wrong

len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_top_losses(9, figsize=(15,11))

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=2)

learn.load('stage_1')

learn.unfreeze() #trains on the whole model

learn.fit_one_cycle(4)

learn.lr_find() #this figures out what is the fastest I can train this NN  

learn.recorder.plot()

learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(3e-05, 1e-05)) # 5% error rate, this says train the first learning layer at 3e-05 and the last layer at 1e-05

learn.fit_one_cycle(2, max_lr=slice(5e-05, 1e-05)) # 4% error rate

learn.fit_one_cycle(2, max_lr=slice(4.5e-05, 8.8e-04)) # 6% error rate

