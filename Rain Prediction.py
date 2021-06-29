#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Loading Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

# ## Loading DataSet

# In[2]:


df=pd.read_csv("C:\\Users\\Lenovo\\Downloads\\testset.csv")


# ## Exploratory Data Analysis

# In[3]:


df.head()


# * All the features that are present in the dataset are listed below

# In[4]:


df.columns


# * Removing space from start and end

# In[5]:


df.columns = map(lambda x:x.strip() , df.columns)


# In[6]:


df.columns


# * Count of different weather conditions from past twenty years in hyderabad is listed below

# In[7]:


df._conds.value_counts()


# * Most common weather conditions in Hyderabad

# In[8]:


plt.figure(figsize=(9,6))
df._conds.value_counts().head(15).plot(kind='bar')
plt.title("Frequent Weather Conditions in Hyderabad")
plt.plot()


# * Least common weather conditions in Hyderabad

# In[9]:


plt.figure(figsize=(9,6))
df._conds.value_counts(ascending=True).head(15).plot(kind='bar')
plt.title("Least Frequent Weather Conditions in Hyderabad")
plt.plot()


# * Wind Direction in Hyderabad

# In[10]:


df._wdire.value_counts()


# In[11]:


plt.figure(figsize=(9, 6));
plt.title("Common wind direction in Hyderabad");
df._wdire.value_counts().plot(kind="bar");
plt.plot();


# * Average Temperature in Hyderabad

# In[12]:


print("Average Temperature : ",round(df._tempm.mean(axis=0),2))


# * Extracting Month and Year for all given data points

# In[13]:


def extract_year(value):
    return (value[0:4])

def extract_month(value):
    return (value[4:6])


# In[14]:


df["year"]=df["datetime_utc"].apply(lambda x:extract_year(x))
df["month"]=df["datetime_utc"].apply(lambda x:extract_month(x))


# In[15]:


df.head()


# * Year range in the dataset

# In[16]:


print("Year range:",df.year.min(),",",df.year.max())


# * Count of number of instances of each year

# In[17]:


df.year.value_counts(ascending=False)


# In[18]:


df1=df.groupby(["year","_rain"]).size()


# * Number of instances when it rained and not rained is stored in df1

# In[19]:


df1.head()


# * Plot for each year vs no of instaces rained and no of instances not rained

# In[20]:


df1.plot(figsize=(15,10),kind='bar',x='year',y='no of sessions')


# * Temperature Mean for each year

# In[21]:


df.groupby("year")._tempm.mean()


# * Humidity Mean for each year

# In[22]:


df.groupby("year")._hum.mean()


# * Pressure Mean for each year

# In[23]:


df.groupby("year")._pressurem.mean()


# ## Handling Missing Data

# * Count of null values in each column/feature

# In[24]:


df.isnull().sum()


# In[25]:


df.columns


# In[26]:


df_filtered = df[['datetime_utc', '_conds', '_dewptm', '_fog', '_hail','_hum', '_pressurem', '_rain', '_snow', '_tempm','_thunder', '_tornado', '_vism', '_wdird', '_wdire', '_wspdm', 'year', "month"]]


# * Checking for null values in dew feature

# In[27]:


df_filtered[df_filtered._dewptm.isnull()]


# * Replacing each null value of dewptm with the mean value of dewptm for that year

# In[28]:


for index,row in df_filtered[df_filtered._dewptm.isnull()].iterrows():
    mean=df_filtered[df_filtered["year"]==row["year"]]._dewptm.mean()
    df_filtered.at[index,"_dewptm"]=mean


# * Now there are no null values for dewptm

# In[29]:


df_filtered[df_filtered._dewptm.isnull()]


# In[30]:


df_filtered.isnull().sum()


# * Checking for null values in humidity

# In[31]:


df_filtered[df_filtered._hum.isnull()]


# * Replacing each null value of hum with the mean value of hum for that year

# In[32]:


for index,row in df_filtered[df_filtered._hum.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._hum.mean()
    df_filtered.at[index, "_hum"] = mean_val


# In[33]:


df_filtered[df_filtered._hum.isnull()]


# In[34]:


df_filtered.isnull().sum()


# * Checking for null values in pressure feature/column

# In[35]:


df_filtered[df_filtered._pressurem.isnull()]


# In[36]:


df_filtered.head()


# In[37]:


df_filtered._pressurem.replace(-9999.0,np.nan,inplace=True)


# In[38]:


df_filtered.head()


# In[39]:


df_filtered._pressurem.isnull().sum()


# * Replacing each null value of pressure with the mean value of pressure for that year

# In[40]:


for index,row in df_filtered[df_filtered._pressurem.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._pressurem.mean()
    df_filtered.at[index, "_pressurem"] = mean_val


# In[41]:


df_filtered.isnull().sum()


# * Replacing each null value of temp with the mean value of temp for that year

# In[42]:


for i,row in df_filtered[df_filtered._tempm.isnull()].iterrows():
    mean = df_filtered[df_filtered["year"] == row["year"]]._tempm.mean()
    df_filtered.at[i, "_tempm"] = mean


# * Replacing each null value of vis with the mean value of vis for that year

# In[43]:


for i,row in df_filtered[df_filtered._vism.isnull()].iterrows():
    mean = df_filtered[df_filtered["year"] == row["year"]]._vism.mean()
    df_filtered.at[i, "_vism"] = mean


# * Replacing each null value of wind with the mean value of wind for that year

# In[44]:


for i,row in df_filtered[df_filtered._wdird.isnull()].iterrows():
    mean = df_filtered[df_filtered["year"] == row["year"]]._wdird.mean()
    df_filtered.at[i, "_wdird"] = mean


# * Replacing each null value of windspeed with the mean value of windspeed for that year

# In[45]:


for i,row in df_filtered[df_filtered._wspdm.isnull()].iterrows():
    mean = df_filtered[df_filtered["year"] == row["year"]]._wspdm.mean()
    df_filtered.at[i, "_wspdm"] = mean


# * Replacing each null value of wind direction with the most frequent values of wind direction for that year

# In[46]:


for index,row in df_filtered[df_filtered._wdire.isnull()].iterrows():
    most_frequent = df_filtered[df_filtered["year"] == row["year"]]._wdire.value_counts().idxmax()
    df_filtered.at[index, "_wdire"] = most_frequent


# * Replacing each null value of weather condition with the most frequent values of weather condition for that year

# In[47]:


for index,row in df_filtered[df_filtered._conds.isnull()].iterrows():
    most_frequent = df_filtered[df_filtered["year"] == row["year"]]._conds.value_counts().idxmax()
    df_filtered.at[index, "_conds"] = most_frequent


# In[48]:


df_filtered.isnull().sum()


# In[49]:


df_filtered.dtypes


# In[50]:


pd.crosstab( index=df_filtered['year'], columns=df_filtered['month'])


# * Year wise number of instances when it rained and not rained

# In[51]:


df2=df_filtered.groupby(["year","_rain"]).size()
print(df2)


# In[52]:


df2.plot(figsize=(15,10),kind='bar')


# * Correlation of features

# In[53]:


print(df_filtered.corr())


# * Correlation Matrix

# In[54]:


plt.figure(figsize=(15,10))
sb.heatmap(df_filtered.corr(), cmap="YlGnBu", annot=True)


# ## Encoding and Target DataFrames

# In[55]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[56]:


df_filtered.columns


# In[57]:


feature_columns = ['_wdire', '_dewptm', '_fog', '_hail', '_hum',
       '_pressurem', '_rain', '_snow', '_tempm', '_thunder', '_tornado',
       '_vism', '_wdird', '_wspdm', 'year', 'month', '_conds']


# In[58]:


df_final=df_filtered[feature_columns]


# In[59]:


df_final.head()


# In[60]:


df_final.dtypes


# In[61]:


df_final._wdire.value_counts()


# In[62]:


wdire_dummies=pd.get_dummies(df_final["_wdire"])
print(wdire_dummies)


# In[63]:


df_final=pd.concat([wdire_dummies,df_final],axis=1)


# In[64]:


df_final.head()


# In[65]:


df_final.columns


# In[66]:


df_final.drop("_wdire",inplace=True,axis=1)


# In[67]:


df_final.columns


# In[68]:


plt.figure(figsize=(25,15))
sb.heatmap(df_final.corr(), cmap="YlGnBu", annot=True)


# In[69]:


df_final.columns


# In[70]:


selected_columns=df_final[["_dewptm","_hum","_thunder","_wspdm","_tempm","_pressurem","_snow"]]


# In[71]:


out_put=df_final[["_rain"]]


# In[72]:


selected_columns


# In[73]:


out_put


# In[74]:


y_rain=out_put.iloc[:,:].values
y_rain.shape


# In[75]:


x_test_selected=selected_columns.iloc[:,:].values


# In[76]:


x_test_selected.shape


# In[77]:


df_final.head(10)


# In[78]:


X=df_final.iloc[:,0:-1].values
X.shape


# In[79]:


print(X)


# In[80]:


Y=df_final.iloc[:,-1].values
Y.shape


# In[81]:


print(Y)


# In[82]:


label_encoder= LabelEncoder()


# In[83]:


y=label_encoder.fit_transform(Y)


# In[84]:


list(label_encoder.classes_)


# In[85]:


y.shape


# In[86]:


np.unique(y)


# In[ ]:





# In[ ]:





# In[87]:


print(y)


# ## Train and Test Split

# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)


# In[90]:


print("Shape of X_train", x_train.shape)
print("Shape of X_test", x_test.shape)
print("Shape of y_train", y_train.shape)
print("Shape of y_test", y_test.shape)


# In[91]:


x_train1,x_test1,y_train1,y_test1 = train_test_split(x_test_selected,y_rain,test_size=.25,random_state=0)


# In[92]:


print("Shape of X_train", x_train1.shape)
print("Shape of X_test", x_test1.shape)
print("Shape of y_train", y_train1.shape)
print("Shape of y_test", y_test1.shape)


# # Model Creation and Training

# ## Decision Tree Classifier

# In[93]:


from sklearn.tree import DecisionTreeClassifier


# In[94]:



#clf=  DecisionTreeClassifier(criterion="entropy",random_state=0)


# In[95]:


#model1 = clf.fit(x_train,y_train)


# In[96]:


#y_pred=clf.predict(x_test)
#y_pred


# In[97]:


#list(label_encoder.inverse_transform(y_pred))


# In[98]:


#list(label_encoder.inverse_transform(y_test))


# ### Accuracy 

# In[99]:


from sklearn import metrics


# In[100]:


#print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# ## Decision Tree Classifier

# In[101]:


clf2=  DecisionTreeClassifier(criterion="entropy",random_state=0)
model1 = clf2.fit(x_train1,y_train1)





# In[150]:


y_pred1=clf2.predict(x_test1)
y_pred1


# ### Accuracy 

# In[103]:


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test1, y_pred1))


# ## Logistic Regression

# In[104]:


#from sklearn.linear_model import LogisticRegression


# In[106]:


#lr = LogisticRegression()


# In[109]:


#lr.fit(x_train1,y_train1)


# In[114]:


#y_pred2=lr.predict(x_test1)
#y_pred2


# ### Accuracy 

# In[115]:


#print("LogisticRegression's Accuracy: ", metrics.accuracy_score(y_test1, y_pred2))


# ## kNN

# In[116]:


#from sklearn.neighbors import KNeighborsClassifier


# In[117]:


#knn = KNeighborsClassifier(n_neighbors=3)


# In[118]:


#knn.fit(x_train1,y_train1)


# In[119]:


#y_pred_knn=knn.predict(x_test1)
#y_pred_knn


# ### Accuracy 

# In[120]:


#print("kNN's Accuracy: ", metrics.accuracy_score(y_test1, y_pred_knn))


# ## RandomForest 

# In[121]:


#from sklearn.ensemble import RandomForestClassifier


# In[122]:


#rf = RandomForestClassifier()


# In[123]:


#rf.fit(x_train1,y_train1)


# In[124]:


#y_pred_random = rf.predict(x_test1)
#y_pred_random


# ### Accuracy 

# In[125]:


#print("Random Forest's Accuracy: ", metrics.accuracy_score(y_test1, y_pred_random))


pickle.dump(clf2, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))



