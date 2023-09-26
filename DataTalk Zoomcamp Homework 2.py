#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression


# In[4]:


df = pd.read_csv('EDA.csv')


# In[6]:


df.describe()


# In[8]:


df.columns


# In[9]:


df.median_house_value.nunique


# In[29]:


dfh=df[(df['ocean_proximity']=='<1H OCEAN') | (df['ocean_proximity']=='INLAND')]


# In[30]:


print(dfh)


# In[31]:


dfh.describe()


# In[33]:


dfh.isnull().sum()


# In[37]:


dfh.population.describe()


# In[42]:


n = len(dfh)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test


# In[43]:


n


# In[54]:


n_val, n_test, n_train


# In[139]:


dfh.iloc[:10]


# In[140]:


dfh_train = dfh.iloc[n_train:]
dfh_val = dfh.iloc[n_train:n_train+n_val]
dfh_test = dfh.iloc[n_val+n_train:]


# In[141]:


np.arange(n)


# In[142]:


idx = np.arange(n)


# In[143]:


np.random.seed(42)
np.random.shuffle(idx)


# In[144]:


idx


# In[145]:


idx[n_train:]


# In[146]:


dfh_train = df.iloc[idx[:n_train]]
dfh_val = df.iloc[idx[n_train:n_train+n_val]]
dfh_test = df.iloc[idx[n_val+n_train:]]


# In[147]:


dfh.head()

dfh_train = dfh_train.reset_index(drop=True)
dfh_test = dfh_test.reset_index(drop=True)
dfh_val = dfh_val.reset_index(drop=True)


# In[148]:


len(dfh_train), len(dfh_val), len(dfh_test)


# In[149]:


y_train = np.log1p(dfh_train.median_house_value.values)
y_val = np.log1p(dfh_val.median_house_value.values)
y_test = np.log1p(dfh_test.median_house_value.values)


# In[229]:


dfh.isnull().sum()


# In[230]:


dfh['total_bedrooms'].fillna(0)


# In[244]:


dfh_train.iloc[10]


# In[247]:


dfh_train


# In[246]:


dfh.columns


# In[251]:


base = ['housing_median_age', 'total_bedrooms', 'population', 'households']

X_train = df_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)

