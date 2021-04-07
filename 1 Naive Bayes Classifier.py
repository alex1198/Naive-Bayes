#!/usr/bin/env python
# coding: utf-8

# ## Naive Bayes Classifier

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("car_data.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# ### Converting Datavalues in Labels

# In[5]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in df.columns:
    df[c] = le.fit_transform(df[c])
df.head(17)


# ### Naive Bayes m - estimate version:-

# In[6]:


'''
this function will return list of class's probabilities, in this dataset there is four probabilies store in list.
'''
def prior(df, y):
    target = list(df[y].unique())
    prior = list()
    for l in target:
#         print(len(df[df[y] == l]))
        prior.append(len(df[df[y]==l])/len(df))
    return prior


# In[7]:


def cal_likelihood(df, col_name, col_val, y, label):
    df = df[df[y] == label]
    probability_x_given_y = len(df[df[col_name] == col_val]) / len(df) # label wise filter the rows and calculate its probility of x|y
    return probability_x_given_y


# In[8]:


def naive_bayes_m_estimate(df, x, y): 
    m = 3
    attrs = list(df.columns)[:-1]  # gather columns names    
    p = prior(df, y)               # call function prior to calculate it 
    y_pred = list() 
    for data in x:                 # loop over every data sample
        # calculate likelihoood
        labels = sorted(list(df[y].unique()))
        likelihood = [1,1,1,1]
        length_labels = len(labels)
        for b in range(length_labels): # for example label = unacc
            for a in range(len(attrs)): # 0,1,2
                likelihood[b] = likelihood[b] * cal_likelihood(df, attrs[a], data[a], y, labels[b])   # calculate the likelihood
        # calculate posterior probability 
        posterior_prob = [1,1,1,1]        
        for c in range(length_labels):
            posterior_prob[c] = (likelihood[c] * (m*(1/p[c]))) / (len(y) + m) # calculate posterior probability with m-estimate
        y_pred.append(np.argmax(posterior_prob))
    return np.array(y_pred)


# ### Calculate the Accuracy of actual and predicated Y:

# In[9]:


def accuracy(y_test, y_pred): 
    l = len(y_test)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(l):
        if y_test[i] == 0:
            if y_pred[i] == 0:
                TN += 1
            else:
                FP += 1
        else:
            if y_pred[i] == 0:
                FN += 1
            else:
                TP += 1
                
    accuracy = ((TP + TN) / (TP + TN + FP + FN )) * 100
    return accuracy             


# In[10]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state = 41)
x_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values
y_pred = naive_bayes_m_estimate(train, x=x_test, y="Acceptability")


# In[11]:


result = accuracy(y_test, y_pred)


# In[12]:


print(result)


# In[13]:


train.to_csv("train.csv")


# In[14]:


test.to_csv("test.csv")


# In[ ]:




