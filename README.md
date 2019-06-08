# Fastai

## Deep Learning GK
1. Download the dataset
2. Take a look at what is in the dataset in terms of files and folders. 
  2.1 We need to find the labels now (Labels in ML are things we are trying to predict).
  2.2 To build a DL model, all you need are pictures (files containing images) and some labels. 
3. Use ImageDataBunch for this^. DataBunch will surely have your training data, later on contain the validation data and can even contain the tesing data.
4. Take a look at your data by doing data.show_batch().
5. Start training the model, using learner.
6. See if you can pre-train your model (for images, this will be helpful, for tabular data, barely any pre-trained).
7. Make sure you don't overfit (where the model only recognizes particular images), use validation set to know if you are overfitting or not.
8. We use validation set instead for this^, set of images that your model does not get to look at and always have the metrics printed for the validation set.  
9. Output your results using fit_one_cycle.

## Lesson 1: Image Classification
Classifying the different breeds of dogs and cats using fine-grained classification methods. 

## Lesson 2: Data Cleaning and Production
Making my own image classification model by using my own data. I want to classify between the two different kinds of Indian classical dances: Kathak and Bharatnatyam. 

## Lesson 4: Tabular Data
