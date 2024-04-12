#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('whitegrid')


# In[2]:


df = pd.read_csv('CarPrice_Assignment.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.sample(10)


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df = df.drop('car_ID',axis=1)


# In[10]:


df


# In[11]:


df.corr()


# In[12]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True , cmap = 'YlGnBu')


# In[13]:


df.columns


# In[14]:


df.symboling.value_counts()


# In[15]:


df.CarName.value_counts()


# In[16]:


df.fueltype.value_counts()


# In[17]:


df.aspiration.value_counts()


# In[18]:


df.doornumber.value_counts()


# In[19]:


df.carbody.value_counts()


# In[20]:


df.drivewheel.value_counts()


# In[21]:


df.enginelocation.value_counts()


# In[22]:


df.enginetype.value_counts()


# In[23]:


df.cylindernumber.value_counts()


# In[24]:


df.fuelsystem.value_counts()


# In[25]:


categorical_cols = df.select_dtypes(include = ['object'])
categorical_cols.head()


# In[26]:


numerical_cols = df.select_dtypes(include = ['float64','int64'])
numerical_cols.head()


# In[27]:


plt.figure(figsize = (20,12))
plt.subplot(3,3,1)
sns.boxplot(x = 'fueltype', y = 'price', data = df)
plt.subplot(3,3,2)
sns.boxplot(x = 'aspiration', y = 'price', data = df)
plt.subplot(3,3,3)
sns.boxplot(x = 'carbody', y = 'price', data = df)
plt.subplot(3,3,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = df)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = df)
plt.subplot(3,3,6)
sns.boxplot(x = 'enginetype', y = 'price', data = df)
plt.subplot(3,3,7)
sns.boxplot(x = 'fuelsystem', y = 'price', data = df)


# In[28]:


df[(df['fueltype'] == 'gas')&(df['price']>30000)]


# In[ ]:





# In[29]:


plt.scatter(df['wheelbase'],df['price'])
plt.xlabel('wheelbase')
plt.ylabel('price')
plt.show()


# In[30]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(14,4))
ax1.scatter(df['symboling'],df['price'])
ax1.set_title('price vs symboling')
ax2.scatter(df['wheelbase'],df['price'])
ax2.set_title('price vs wheelbase')
ax3.scatter(df['carlength'],df['price'])
ax3.set_title('price vs carlength')
plt.show()


# In[31]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(14,4))
ax1.scatter(df['carwidth'],df['price'])
ax1.set_title('price vs carwidth')
ax2.scatter(df['carheight'],df['price'])
ax2.set_title('price vs carheight')
ax3.scatter(df['curbweight'],df['price'])
ax3.set_title('price vs curbweight')
plt.show()


# In[32]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(14,4))
ax1.scatter(df['enginesize'],df['price'])
ax1.set_title('price vs enginesize')
ax2.scatter(df['boreratio'],df['price'])
ax2.set_title('price vs boreratio')
ax3.scatter(df['stroke'],df['price'])
ax3.set_title('price vs stroke')
plt.show()


# In[33]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(14,4))
ax1.scatter(df['compressionratio'],df['price'])
ax1.set_title('price vs compressionratio')
ax2.scatter(df['horsepower'],df['price'])
ax2.set_title('price vs horsepower')
ax3.scatter(df['peakrpm'],df['price'])
ax3.set_title('pric vs peakrpm')
plt.show()


# In[34]:


f,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(14,4))
ax1.scatter(df['citympg'],df['price'])
ax1.set_title('price vs citympg')
ax2.scatter(df['highwaympg'],df['price'])
ax2.set_title('price vs highwaympg')
plt.show()


# In[35]:


from sklearn.preprocessing import LabelEncoder


# In[36]:


le = LabelEncoder()

var_mod = df.select_dtypes(include='object').columns


for i in var_mod:
    df[i] = le.fit_transform(df[i])


# In[37]:


df


# In[38]:


X = df.drop(['price'],axis=1)


# In[39]:


X


# In[40]:


y = df['price']


# In[41]:


y


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[44]:


X_train.shape


# In[45]:


X_test.shape


# In[51]:


from sklearn.tree import DecisionTreeRegressor


# In[52]:


model = DecisionTreeRegressor()


# In[54]:


model.fit(X_train,y_train)
DecisionTreeRegressor()


# In[55]:


model.score(X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




