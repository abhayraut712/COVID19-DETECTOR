#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# In[50]:


df = pd.read_csv("Datasets/compiledDataset.csv")


# In[51]:


df.drop("Unnamed: 0",axis=1,inplace=True)


# In[52]:


df.head()


# In[ ]:





# In[53]:


corrDF = df.corr()
corrDF


# In[ ]:





# In[ ]:





# In[ ]:





# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


X = df.drop("covid_19",axis=1)
y = df["covid_19"]


# In[56]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


# # Benchmarking with a Simple Logistic Model

# In[57]:


from sklearn.linear_model import LogisticRegression


# In[58]:


log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)


# In[59]:


from sklearn.metrics import classification_report
print("-"*30)

y_pred = log_reg.predict(X_train)
print("Logistic Regression - Training set")
print("-"*30)
print(classification_report(y_train, y_pred))

print("-"*30)

y_pred = log_reg.predict(X_test)
print("Logistic Regression - Test set")
print("-"*30)
print(classification_report(y_test, y_pred))


# # SVM Classifier

# In[60]:


from sklearn.svm import SVC


# In[61]:


svc = SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)


# In[62]:


from sklearn.metrics import classification_report
print("-"*30)

y_pred = svc.predict(X_train)
print("SVM - Training set")
print("-"*30)
print(classification_report(y_train, y_pred))

print("-"*30)

y_pred = svc.predict(X_test)
print("SVM - Test set")
print("-"*30)
print(classification_report(y_test, y_pred))


# In[63]:


accuracy_score(y_test, y_pred)


# In[64]:


df.columns


# In[65]:


# save the model to disk
filename = './Models/svmModel.sav'
pickle.dump(svc, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
# classify = loaded_model.predict("get a chance to win $10,000 by clicking this link")
print(result)


# # Standardize the Data

# In[66]:


from sklearn.preprocessing import StandardScaler


# In[67]:


features = X.columns
feat = X

# Separating out the features
x = feat.loc[:, features].values

# Separating out the target
y = df.loc[:,['covid_19']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)


# # PCA Projection to 2D

# In[68]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[69]:


principalDf.head()


# In[70]:


finalDf = pd.concat([principalDf, df[['covid_19']]], axis = 1)


# In[71]:


plt.figure(figsize=(15, 8))
label = np.array(df.covid_19)
for i in set(label):
    d = finalDf[label == i]
    plt.scatter(x=d['principal component 1'], y=d['principal component 2'], label=i)

plt.xlabel("1st_principal")
plt.ylabel("2nd_principal")
plt.title("Feature Representation in 2D Plot")
plt.legend()


# In[ ]:





# In[72]:


inpDF = pd.DataFrame(np.nan, index=[0], columns=['Clickbaits'])
inpDF


# In[ ]:





# In[ ]:




