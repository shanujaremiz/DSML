#!/usr/bin/env python
# coding: utf-8

# In[41]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt


# In[42]:


(X_train, y_train), (X_valid, y_valid) = mnist.load_data()


# In[43]:


type(X_train)


# In[44]:


X_train.shape


# In[45]:


y_train.shape


# In[46]:


y_train[0:20]


# In[47]:


plt.figure(figsize=(5,5))
for k in range(20):
   plt.subplot(10, 2, k+1)
   plt.imshow(X_train[k],cmap='Greys')
   plt.axis('off')
plt.tight_layout()
plt.show()


# In[48]:


X_valid.shape


# In[49]:


y_valid.shape


# In[50]:


plt.imshow(X_valid[0], cmap='Greys')


# In[51]:


X_valid[0]


# In[52]:


#we convert the labels to one hot representation
from keras import utils as np_utils
n_classes = 10

y_train = keras.utils.np_utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.np_utils.to_categorical(y_valid, n_classes)


# In[53]:


y_valid[0]


# In[54]:


#preprocess data


# In[55]:


X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')


# In[56]:


X_train /= 255
X_valid /= 255


# In[57]:


X_valid[0]


# In[58]:


model = Sequential()


# In[59]:


model.add(Dense(64, activation='sigmoid', input_shape=(784,)))


# In[60]:


model.add(Dense(10, activation='softmax'))


# In[61]:


model.summary()


# In[62]:


model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01),metrics=['accuracy'])


# In[63]:


history=model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1)


# In[ ]:




