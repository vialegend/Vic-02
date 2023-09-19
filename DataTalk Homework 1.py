#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


# In[3]:


data = pd.read_csv("raw.githubusercontent.com_alexeygrigorev_datasets_master_housing.csv")


# In[4]:


data.head()


# In[8]:


data.shape


# In[12]:


data.info()


# In[13]:


data.describe()


# In[15]:


data.isnull().sum()


# In[17]:


data.ocean_proximity


# In[19]:


data.ocean_proximity.describe()


# In[22]:


data.median_house_value.describe().round(0)


# In[24]:


data.groupby('ocean_proximity').median_house_value.mean().round(0)


# In[29]:


data.total_bedrooms.describe().round(0)


# In[47]:


data['total_bedrooms'] = data['total_bedrooms'].fillna(538)


# In[49]:


data.isnull().sum()


# In[50]:


data.describe().round(3)


# In[58]:


data[['housing_median_age', 'total_rooms', 'total_bedrooms']]


# In[71]:


data.ocean_proximity.nunique


# In[74]:


data.ocean_proximity.describe()


# In[91]:


df = pd.DataFrame(data)

df.head()


# In[96]:


X = df.loc[df['ocean_proximity']=='ISLAND',['housing_median_age','total_rooms','total_bedrooms']].values


# In[108]:


X


# In[114]:


XT = np.transpose(X)


# In[115]:


XT


# In[123]:


print(X.shape, X.T.shape)


# In[125]:


XTX = XT.dot(X)


# In[127]:


XTX


# In[128]:


XTX_inv = np.linalg.inv(XTX)


# In[129]:


XTX_inv


# In[130]:


Y = [950, 1300, 800, 1000, 1300]


# In[131]:


Y


# In[134]:


w = XTX_inv.dot(X.T).dot(Y) 


# In[135]:


w


# In[ ]:




