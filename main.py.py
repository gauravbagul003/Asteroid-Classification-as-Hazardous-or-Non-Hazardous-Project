#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


df = pd.read_csv('D:/nasa.csv')


# In[21]:


df.head()


# In[22]:


df.shape


# In[23]:


df.info()


# In[24]:


df = df.drop(['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date', 'Epoch Date Close Approach', 'Orbit Determination Date'], axis = 1)
df.head()


# In[25]:


hazardous_labels = pd.get_dummies(df['Hazardous'])
hazardous_labels


# In[26]:


df = pd.concat([df, hazardous_labels], axis = 1)
df.head()


# In[27]:


df = df.drop(['Hazardous'], axis = 1)
df.head()


# In[28]:


df.info()


# In[29]:


df['Orbiting Body'].value_counts()


# In[30]:


df['Equinox'].value_counts()


# In[33]:


df = df.drop(['Orbiting Body', 'Equinox'], axis = 1)


# In[34]:


plt.figure(figsize = (20,20))
sns.heatmap(df.corr(), annot = True)


# In[35]:


df = df.drop(['Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)', 'Miss Dist.(kilometers)', 'Miss Dist.(miles)'], axis = 1)
df.head()


# In[36]:


plt.figure(figsize = (20,20))
sns.heatmap(df.corr(), annot = True)


# In[37]:


df.drop([False], axis = 1, inplace = True)


# In[38]:


df.head()


# In[39]:


df.describe()


# In[40]:


x = df.drop([True], axis = 1)
y = df[True].astype(int)


# In[42]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.3)


# In[44]:


from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance

xbg_model = XGBClassifier()
xbg_model.fit(x_train, y_train)
plot_importance(xbg_model)
pyplot.show()


# In[52]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
predictions = xbg_model.predict(x_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+ '%')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# In[ ]:




