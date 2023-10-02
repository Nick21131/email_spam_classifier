#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv("mail_data.csv")


# In[3]:


print(df)


# In[4]:


data=df.where((pd.notnull(df)),"")


# In[5]:


data.head(7)


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


data.loc[data["Category"]=="spam","Category",]=0
data.loc[data["Category"]=="ham","Category",]=1


# In[9]:


X=data["Message"]
Y=data["Category"]


# In[10]:


print(X)
print(Y)


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[30]:


print(X.shape)
print(x_train.shape)
print(x_test.shape)
print(Y.shape)
print(y_train.shape)
print(y_test.shape)


# In[31]:


feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)


# In[32]:


x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)


# In[33]:


y_train = y_train.astype(int)
y_test = y_test.astype(int)


# In[34]:


print(x_train)


# In[35]:


print(x_train_features)


# In[36]:


model=LogisticRegression()


# In[37]:


model.fit(x_train_features, y_train)


# In[38]:


prediction_on_training_data=model.predict(x_train_features)
accuracy_on_training_data=accuracy_score(y_train,prediction_on_training_data)


# In[39]:


print("Acc on training data:",accuracy_on_training_data)


# In[40]:


prediction_on_test_data = model.predict(x_test_features)


# In[42]:


accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print("Accuracy on test data:", accuracy_on_test_data)


# In[53]:


input_your_mail=["you won 100000$ on a lottery. please contact us at 99989899878"]
input_data_features=feature_extraction.transform(input_your_mail)
prediction=model.predict(input_data_features)
print(prediction)
if(prediction==1):
    print("ham mail")
else:
    print("spam mail")

