#!/usr/bin/env python
# coding: utf-8

#  Task 1 Iris Classification

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# # Loading the dataset

# In[13]:


df = pd.read_csv("C:\\Users\\This PC\\Downloads\\IRIS.csv")
df.head()


# In[14]:


#to display status about data
df.describe()


# In[65]:


# to basic info about datatype
df.info()


# # Preprocessing the dataset

# In[25]:


# check for null values
df.isnull().sum()


# # Exploratory Data Analysis

# In[29]:


#histograms
df['sepal_length'].hist()


# In[32]:


df['sepal_width'].hist()


# In[33]:


df['petal_length'].hist()


# In[34]:


df['petal_width'].hist()


# In[35]:


#scatterplot
colors = ['green','yellow','blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[37]:


for i in range(3):
    x = df[df['species']== species[i]]
    plt.scatter(x['sepal_length'], x['sepal_width'], c = colors[i], label = species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
    


# In[38]:


for i in range(3):
    x = df[df['species']== species[i]]
    plt.scatter(x['petal_length'], x['petal_width'], c = colors[i], label = species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[39]:


for i in range(3):
    x = df[df['species']== species[i]]
    plt.scatter(x['sepal_length'], x['petal_length'], c = colors[i], label = species[i])
plt.xlabel("sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[41]:


for i in range(3):
    x = df[df['species']== species[i]]
    plt.scatter(x['sepal_length'], x['petal_width'], c = colors[i], label = species[i])
plt.xlabel("sepal Length")
plt.ylabel("Petal Width")
plt.legend()


# #  Correlation Matrix

# In[47]:


df.corr()


# In[49]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# # Label Encoder

# In[50]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[51]:


df['species'] = le.fit_transform(df['species'])
df.head()


# #  Model Training

# In[52]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['species'])
Y = df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[53]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[54]:


# model training
model.fit(x_train, y_train)


# In[55]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[56]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[57]:


model.fit(x_train, y_train)


# In[58]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[59]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[60]:


model.fit(x_train, y_train)


# In[61]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[62]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','KNeighborsClassifier','Decision Tree'],
    'Score': [97.777,95.555,97.777]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:




