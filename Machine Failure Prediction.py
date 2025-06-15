#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\Users\Sanjana\Downloads\predictive_maintenance.csv")


# In[3]:


df.head()
#The dataset consists of 10 000 data points stored as rows with 14 features in columns

# UID: unique identifier ranging from 1 to 10000
# productID: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number
# air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
# process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
# rotational speed [rpm]: calculated from powepower of 2860 W, overlaid with a normally distributed noise
# torque [Nm]: torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
# tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a
# 'machine failure' label that indicates, whether the machine has failed in this particular data point for any of the following failure modes are true.
# Important : There are two Targets - Do not make the mistake of using one of them as feature, as it will lead to leakage.
# Target : Failure or Not
# Failure Type : Type of Failure


# In[4]:


df.describe()


# In[5]:


df.drop_duplicates()


# In[6]:


upp=df['Rotational speed [rpm]'].quantile(0.95)


# In[7]:


low=df['Rotational speed [rpm]'].quantile(0.05)


# In[8]:


df=df[(df['Rotational speed [rpm]']<upp)&(df['Rotational speed [rpm]']>low)]


# In[9]:


df.describe()


# In[10]:


df['Temp difference']=df['Process temperature [K]']-df['Air temperature [K]']
df.describe()


# In[11]:


sns.countplot(data=df,x="Target")
plt.show()


# In[12]:


plt.figure(figsize=(15,5))
sns.countplot(data=df,x="Failure Type")
plt.show()


# In[13]:


sns.barplot(data=df,x="UDI",y="Failure Type",hue="Target")


# In[14]:


exp=df[(df['Target']==1)&(df['Failure Type']=='No Failure')].index


# In[15]:


df1=df.drop(exp)
df1


# In[16]:


sns.barplot(data=df1,x="UDI",y="Failure Type",hue="Target")


# In[17]:


corr_matrix=df1.corr()


# In[18]:


sns.heatmap(corr_matrix,annot=True)


# #### Target- torque is the best predictor, then tool wear and then air temp and process temp

# In[32]:


plt.figure(figsize=(20,3))

plt.subplot(1,5,1)
plt.hist(df1['Torque [Nm]'], bins=20,color='green',edgecolor='black',rwidth=0.8,density=True)

plt.subplot(1,5,2)
plt.hist(df1['Tool wear [min]'], bins=20,color='blue',edgecolor='black',rwidth=0.8,density=True)

plt.subplot(1,5,3)
plt.hist(df1['Air temperature [K]'], bins=20,color='yellow',edgecolor='black',rwidth=0.8,density=True)

plt.subplot(1,5,4)
plt.hist(df1['Process temperature [K]'], bins=20,color='orange',edgecolor='black',rwidth=0.8,density=True)

plt.subplot(1,5,5)
plt.hist(df1['Rotational speed [rpm]'], bins=20,color='red',edgecolor='black',rwidth=0.8,density=True)


# In[47]:


plt.figure(figsize=(20,3))

plt.subplot(1,5,1)
plt.scatter(data=df1,x='Air temperature [K]',y='Process temperature [K]')

plt.subplot(1,5,2)
plt.scatter(data=df1,x='Air temperature [K]',y='Tool wear [min]')

plt.subplot(1,5,3)
plt.scatter(data=df1,x='Air temperature [K]',y='Torque [Nm]')

plt.subplot(1,5,4)
plt.scatter(data=df1,x='Air temperature [K]',y='Rotational speed [rpm]')


# In[48]:


plt.figure(figsize=(20,3))

plt.subplot(1,5,1)
plt.scatter(data=df1,x='Process temperature [K]',y='Air temperature [K]')

plt.subplot(1,5,2)
plt.scatter(data=df1,x='Process temperature [K]',y='Tool wear [min]')

plt.subplot(1,5,3)
plt.scatter(data=df1,x='Process temperature [K]',y='Torque [Nm]')

plt.subplot(1,5,4)
plt.scatter(data=df1,x='Process temperature [K]',y='Rotational speed [rpm]')


# In[49]:


plt.figure(figsize=(20,3))

plt.subplot(1,5,1)
plt.scatter(data=df1,x='Tool wear [min]',y='Air temperature [K]')

plt.subplot(1,5,2)
plt.scatter(data=df1,x='Tool wear [min]',y='Process temperature [K]')

plt.subplot(1,5,3)
plt.scatter(data=df1,x='Tool wear [min]',y='Torque [Nm]')

plt.subplot(1,5,4)
plt.scatter(data=df1,x='Tool wear [min]',y='Rotational speed [rpm]')


# In[50]:


plt.figure(figsize=(20,3))

plt.subplot(1,5,1)
plt.scatter(data=df1,x='Torque [Nm]',y='Air temperature [K]')

plt.subplot(1,5,2)
plt.scatter(data=df1,x='Torque [Nm]',y='Process temperature [K]')

plt.subplot(1,5,3)
plt.scatter(data=df1,x='Torque [Nm]',y='Rotational speed [rpm]')


# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X = df1[["Process temperature [K]", "Air temperature [K]", "Rotational speed [rpm]", "Torque [Nm]","Tool wear [min]"]] #Features
y = df1[["Target"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)

model=DecisionTreeClassifier(max_depth=2)
model.fit(X_train,y_train)

predictions=model.predict(X_test)

acc=accuracy_score(y_test,predictions)
print(acc)


# In[65]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100, random_state=43)

rfc.fit(X_train,y_train)

pred_rfc=rfc.predict(X_test)

rfc_acc=accuracy_score(y_test,pred_rfc)
print(rfc_acc)


# In[66]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, pred_rfc)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
disp.plot(cmap='Blues')

plt.title("Confusion Matrix")
plt.show()


# In[73]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

tree_to_plot = rfc.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()


# In[93]:


from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt

features = [0, 1, 2, 3, 4]  # all 5 features by index

fig, ax = plt.subplots(figsize=(15, 10))
plt.title("Predicted values")
plot_partial_dependence(rfc, 
                        X=df1[["Process temperature [K]", "Air temperature [K]", 
                               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]],
                        features=features, 
                        grid_resolution=50,
                        ax=ax)

plt.tight_layout()
plt.show()


# In[ ]:




