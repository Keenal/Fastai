

from fastai.tabular import *
from fastai.tabular import *
!git clone https://github.com/muellerzr/ML_Useful_Functions.git
from ML_Useful_Functions.Functions import *

"""Tabular data should be in a Pandas `DataFrame`."""

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')

SplitSet??

train, test = SplitSet(df)

dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [FillMissing, Categorify, Normalize]

test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)

data = (TabularList.from_df(train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_rand_pct(.2)
                           .label_from_df(cols=dep_var)
                           .databunch())

data.show_batch(rows=10)

learn = tabular_learner(data, layers=[200,100], metrics=accuracy)

learn.fit(1, 1e-2)

"""## Inference"""

row = test.iloc[0] #inference is when you make prediction

learn.predict(row)

calcHiddenLayer??

calcHiddenLayer(data, 3, 2) #output will be layer sizes

learn = tabular_learner(data, layers=calcHiddenLayer(data, 3, 2), metrics=accuracy)

learn.fit(1) #having understanding of how big that layer size is

"""## Find alpha"""

cat_var, cont_var = feature_importance(learn, cat_names, cont_names)

data = (TabularList.from_df(train, path=path, cat_names=cat_var, cont_names=cont_var, procs=procs)
                           .split_by_rand_pct(.2)
                           .label_from_df(cols=dep_var)
                           .databunch())

learn = tabular_learner(data, layers=calcHiddenLayer(data, 3, 2), metrics=accuracy)

learn.fit_one_cycle(1)



"""## Predictions"""

testData = (TabularList.from_df(test, path=path, cat_names=cat_var, cont_names=cont_var, procs=procs).split_none().label_from_df(cols=dep_var).databunch())

results = learn.validate(testData.train_dl)

acc = float(results[1]) * 100
print("Test accuracy of: " + str(acc))

!PredictTest??

