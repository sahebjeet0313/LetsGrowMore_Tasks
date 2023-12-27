#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv("C:\\Users\\saheb\\Downloads\\iris.csv")


# In[3]:


print(iris.head())


# In[4]:


print(iris.describe())


# In[6]:


import plotly.express as px


# In[7]:


fig1 = px.scatter(iris,x="sepal_width", y="sepal_length", color="species")
fig1.show()


# In[8]:


fig2 = px.scatter(iris,x="petal_width", y = "petal_length", color="species")
fig2.show()


# In[9]:


# now we will use KNN algorithm for classification

x = iris.drop("species",axis=1)
print(x)


# In[10]:


y = iris["species"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)


x_new = np.array([[5.2,2.9,2.8,0.8]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[ ]:




