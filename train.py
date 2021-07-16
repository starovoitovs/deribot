#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from csv import writer
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Input, LSTM, Conv1D, Conv2D, Reshape, Dense, BatchNormalization, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import *
from utils import *


# In[29]:


N_EPOCHS = 30
BATCH_SIZE = 128

# size of the time-series slice used as input to the NN
WINDOW_SIZE = 100

# offset for the rolling windows used for training
OFFSET = 1

# size of the smoothing kernel used to generate labels
KERNEL_SIZE = 50

# labels are quantized into -1, 0, 1 based on this threshold
LABEL_THRESHOLD = 1e-5


# In[13]:


apcs = [f'ap_{i}' for i in range(10)]
avcs = [f'av_{i}' for i in range(10)]
bpcs = [f'bp_{i}' for i in range(10)]
bvcs = [f'bv_{i}' for i in range(10)]

keys = [x for items in zip(apcs, bpcs, avcs, bvcs) for x in items]
features = ['bid_diff_feature_1', 'ask_diff_feature_1']


# # Data

# In[27]:


df = pd.read_csv('_input/data/20210714_182011.csv')[:4000]
df.index = pd.to_datetime(df['timestamp'] * 1000 * 1000)
assert df.index.is_unique


# In[28]:


df = prepare_df1(df)
df = prepare_df2(df, k=KERNEL_SIZE)


# In[16]:


N0, N1 = 0, 800

for label in 'y1', 'y2':

    plt.figure(figsize=(15, 5))

    # colors
    cs = -1 + 1. * (dfX.iloc[N0:N1][label] > -LABEL_THRESHOLD) + 1. * (dfX.iloc[N0:N1]['y1'] > LABEL_THRESHOLD)

    # price
    plt.step(np.arange(N1-N0), dfX.iloc[N0:N1]['mid_price'], c='b')
    add_pcolor_to_plot(plt.gca(), cs)

    # label
    plt.twinx().step(np.arange(N1-N0), dfX.iloc[N0:N1][label], c='r')
    add_pcolor_to_plot(plt.gca(), cs)


# In[17]:


def split(df, offset, window_size):
    
    threshold = 5e-5
    
    I = np.array([x + np.arange(window_size) for x in np.arange(0, len(df) - window_size, offset)])
    
    # X1
    X1 = df[keys].to_numpy()[I]
    X1 = X1[:, :, :, np.newaxis]
    
    # X2
    X2 = df[features].to_numpy()[I]

    # y
    y = dfX.iloc[offset-1+window_size:len(df):offset]['y2'].to_numpy()
    y = -1 + 1. * (y >= -threshold) + 1. * (y >= threshold)
    
    # one-hot encoder
    enc = OneHotEncoder(sparse=False)
    y = enc.fit_transform(y.reshape(-1, 1))

    return X1, X2, y


# In[18]:


X1, X2, y = split(dfX, OFFSET, WINDOW_SIZE)


# In[19]:


X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.15, shuffle=False)


# # Build model

# In[20]:


model = build_model(window_size=WINDOW_SIZE, n_features=len(features))
model.summary()


# In[22]:


cp_callback = ModelCheckpoint(filepath='_output/model.h5', monitor='val_accuracy', save_weights_only=True, save_best_only=True)
history = model.fit([X1_train, X2_train], y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15, callbacks=[cp_callback], verbose=2)


# In[23]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()


# In[24]:


# load and evaluate best model

model.load_weights('_output/model.h5')

loss, accuracy = model.evaluate([X1_test, X2_test], y_test, verbose=0)

print(f"{'loss':15}{loss:7.4f}")
print(f"{'accuracy':15}{accuracy:7.4f}")

y_pred_num = np.argmax(model.predict([X1_test, X2_test]), axis=1)
y_test_num = np.argmax(y_test, axis=1)

precision, recall, fscore, support = precision_recall_fscore_support(y_pred_num, y_test_num)

print(f"{'precision':15}{' '.join(['{:7.2f}'.format(x) for x in precision])}")
print(f"{'recall':15}{' '.join(['{:7.2f}'.format(x) for x in recall])}")
print(f"{'fscore':15}{' '.join(['{:7.2f}'.format(x) for x in fscore])}")
print(f"{'support':15}{' '.join(['{:7}'.format(x) for x in support])}")

