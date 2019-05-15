#from https://www.kaggle.com/poonaml/deep-neural-network-keras-way/data
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
# %matplotlib inline

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", ""]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("train.csv")
print(train.shape)
train.head()

test = pd.read_csv("test.csv")
print(test.shape)
test.head()

X_train = (train.iloc[:, 1:].values).astype('float32')  # all pixel values
y_train = train.iloc[:, 0].values.astype('int32')  # only labels i.e targets digits
X_test = test.values.astype('float32')

print(X_train)
print(y_train)

X_train = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
    plt.show()

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px

from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes

plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10));
plt.show()

