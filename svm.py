#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Import the dataset using Seaborn library
iris=pd.read_csv('IRIS.csv')


# In[6]:


# Checking the dataset
iris.head()


# In[7]:


# Creating a pairplot to visualize the similarities and especially difference between the species
sns.pairplot(data=iris, hue='species', palette='Set2')


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


# Separating the independent variables from dependent variables
x=iris.iloc[:,:-1]
y=iris.iloc[:,4]
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)


# In[10]:


from sklearn.svm import SVC
model=SVC()


# In[11]:


model.fit(x_train, y_train)


# In[12]:


pred=model.predict(x_test)


# In[13]:


#model evaluation
# Importing the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix


# In[14]:


print(confusion_matrix(y_test,pred))


# In[15]:


print(classification_report(y_test, pred))


# In[ ]:




