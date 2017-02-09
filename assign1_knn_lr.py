# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle 

print('import the package')

##you need to write you pickle_file to storage the pickle file

pickle_file = '/home/shaomingguang/shao_data/notMNIST.pickle'
#with open(pickle_file, 'rb') as f:
#    letter_set = pickle.load(f)

f = open(pickle_file)
dict_dataset = pickle.load(f)
f.close()
#print(dict_dataset.viewkeys())
#dict_dataset.viewkeys()

print('let us load the data')

train_data = dict_dataset['train_dataset']
train_label = dict_dataset['train_labels']

test_data = dict_dataset['test_dataset']
test_label = dict_dataset['test_labels']

valid_data = dict_dataset['valid_dataset']
valid_label = dict_dataset['valid_labels']
   
print ("Now tansform our data for sklearn")
from sklearn import linear_model, neighbors
train_data_trans = train_data.reshape(len(train_data),28*28)
test_data_trans = test_data.reshape(len(test_data),28*28)
valid_data_trans = valid_data.reshape(len(valid_data),28*28)

#print("train",train_data_trans.shape,train_label.shape)
#print("test",test_data_trans.shape,test_label.shape)
print(train_data_trans.shape)
print ("Transform OK!")

print ("Let's train the model")

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

print('LogisticRegression score: %f'
      % logistic.fit(train_data_trans, train_label).score(test_data_trans, test_label))

print('KNN score: %f' % knn.fit(train_data_trans, train_label).score(test_data_trans, test_label))
