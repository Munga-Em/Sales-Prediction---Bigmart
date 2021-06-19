#!/usr/bin/env python
# coding: utf-8

# <h>Big Mart Sales Prediction</h>
# 
# Train Source: https://www.kaggle.com/devashish0507/big-mart-sales-prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


Train = pd.read_csv("Desktop/New folder/Train.csv", delimiter=',')
Test = pd.read_csv('Desktop/New folder/Test.csv', delimiter=',')


# In[3]:


Train.info()
print('-----------------------------------------------------------------------------------------------')
Train.describe(include='all').round(2)


# In[4]:


Train.head().round(2)


# <p>There are 8,523 observations, 11 features and target variable, Outlet sales.</p>
# 
# Item weight and outlet size have some missing values. 4 features are numeric, 7 non-numeric and the target variable is also numeric.
# 
# Going by item_identifier, Bigmart has a wide variety of products, with the top product appearing only 10 times. Most customers also prefer low fat items, fruits&vegetables and store OUT027.

# In[5]:


Train['Item_Weight'].hist()
plt.show()


# Null values of the Item_Weight feature are filled with mean values of items grouped by Item_Identifier.

# In[6]:


s = Train.groupby('Item_Identifier')['Item_Weight'].mean()

missing = Train['Item_Weight'].isnull()
Train.loc[missing,'Item_Weight'] = s[Train.loc[missing, 'Item_Identifier']].values

#Fill any remaining values with mean of items grouped by item type

m = Train.groupby('Item_Type')['Item_Weight'].mean()
missing = Train['Item_Weight'].isnull()
Train.loc[missing,'Item_Weight'] = m[Train.loc[missing, 'Item_Type']].values


# In[7]:


Train['Item_Weight'].isnull().any()


# In[8]:


Train['Item_Weight'].hist()
plt.show()


# In[9]:


Train[Train['Outlet_Size'].isnull()]['Outlet_Type'].value_counts()


# In[10]:


Train[Train['Outlet_Type'] == 'Grocery Store']['Outlet_Size'].value_counts()


# In[11]:


Train[Train['Outlet_Type'] == 'Supermarket Type1']['Outlet_Size'].value_counts(normalize=True)


# A good number of outlets(2,410) don't have outlet size details. The outlets are of two types, grocery store and supermarket type1, so the missing values are filled accordingly

# In[12]:


#Fill outlet size where outlet type is grocery store
Train['Outlet_Size'] = np.where(Train['Outlet_Type']=='Grocery Store', 'Small', Train['Outlet_Size'])


#Fill outlet size of remaining supermarket type1 with values in same ratio
Train['Outlet_Size'] = Train['Outlet_Size'].fillna(pd.Series(np.random.choice(['Medium', 'Small', 'High'], 
                                                      p=[0.25, 0.50, 0.25], size=len(Train))))


# In[13]:


#Select all non-numeric columns for further analysis
cols = Train.iloc[:, [2,4,6,8,9,10]]

for i in cols.columns:
    print(pd.value_counts(cols[i]))


# Item_Fat_Content seems to have, only 2 categories, low fat and regular. So 'low fat' and 'LF' are replaced with 'Low Fat' while 'reg' is replaced with 'regular'. Not sure whether fat content in non-food items really guides a buyers decision, so a third category, 'Non Food' is created for those.
# 
# High is replaced with Large in Outlet size.

# In[14]:


Train['Item_Fat_Content'] = Train['Item_Fat_Content'].replace(['low fat', 'LF', 'reg'],['Low Fat', 'Low Fat', 'Regular'])
Train['Item_Fat_Content'] = np.where(Train['Item_Type'] =='Household','Non Food', Train['Item_Fat_Content'])
Train['Item_Fat_Content'] = np.where(Train['Item_Type'] =='Health and Hygiene','Non Food', Train['Item_Fat_Content'])
Train['Item_Fat_Content'] = np.where(Train['Item_Type'] =='Others','Non Food', Train['Item_Fat_Content'])
Train['Outlet_Size'] = Train['Outlet_Size'].replace('High', 'Large')


# In[15]:


Train.isnull().sum()


# Next we check the distribution of each column values

# In[16]:


#Select categorical columns
cols = Train.iloc[:, [2,4,6,8,9,10]]


# In[17]:


fig, axs = plt.subplots(nrows = 3, ncols=2, figsize=(15,15))
fig.subplots_adjust(hspace = .7, wspace=.2)
    
for ax, i in zip(axs.ravel(), cols.columns):
    sns.countplot(x = cols[i], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


# Low fat is still prefered over regular, but OUT027 is just the most frequent in the Train, but not necesarily the most prefered. Only 2 outlets are not performing at its level.
# 
# Fruits & vegetables and snack foods are the top 'sellers' in terms of frequency, maybe because of perishability, or maybe because it's the main business, while sea food has the least fans, maybe coz Bigmart is just not targeting seafood lovers.
# 

# In[18]:


#Numeric values
cols1 = Train.iloc[:, [1,3,5,7,11]]


# In[19]:


fig, axs = plt.subplots(nrows = 3, ncols=2, figsize=(15,15))
fig.subplots_adjust(hspace = .7, wspace=.2)
    
for ax, i in zip(axs.ravel(), cols1.columns):
    cols1[i].hist(bins=10, grid = False, ax=ax)
    ax.set(xlabel=i, ylabel='freq')
plt.show()


# Item_Visibility,the % of total display area of all products in a store, allocated to the particular product, has many values that can be rounded to zero,, which indicates that Bigmart sold many small items or items of little display needs.
# 
# Most items have a maximum retail price of between 100 and 200
# 
# No outlets were opened between 1987 & 1997. Most of the stores were opened at the beginning or immediately after the 10 year break. As part of feature engineering, a column for Outlet age is created, to use instead of establishment year
# 
# Most items have a sales value of below 3,000 for every outlet, probably indicating that they don't stock many pricey items.

# In[20]:


Train[Train['Item_Visibility'].round(2)==0]['Item_Type'].value_counts().plot(kind='barh')
plt.xlabel(xlabel='Count')
plt.ylabel(ylabel='Item Type')

plt.show()


# In[21]:


plt.scatter(Train['Item_MRP'],Train['Item_Outlet_Sales'])
plt.show()


# Items seem to have 4 distinct price groups, and a column is created to reflect that. The groups will be Low, Medium, High, Very High

# <h1> FEATURE ENGINEERING </h1>

# There are 16 different categories for Item_Type which are in similar broader cartegories, so the categories are reduced to 4: Foods, Drinks, Non_foods and Others
# 
# Year of establishment is replaced by age (2013 is the year the dataset relates to).
# Item_Price_Group is created from Item_MRP.
# Categorical features are encoded to numbers (Ordinal: Item_Fat_Content, Outlet_Size, Outlet_Location_Type, Outlet_Type. Nominal: Item_Type, Outlet_Identifier, Item_Identifier)

# In[22]:


Train['Item_Type'] = Train['Item_Type'].replace(['Fruits and Vegetables', 'Snack Foods',
                                                     'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods',
                                                     'Meat', 'Breads', 'Starchy Foods', 'Breakfast', 'Seafood'],
                                                     'Foods')
Train['Item_Type'] = Train['Item_Type'].replace(['Household', 'Health and Hygiene'],
                                                     'Non Foods')
Train['Item_Type'] = Train['Item_Type'].replace(['Soft Drinks', 'Hard Drinks'],
                                                     'Drinks')
Train['Item_Outlet_Qty'] = (Train['Item_Outlet_Sales']/Train['Item_MRP']).round()


# In[23]:


Train['Outlet_Age'] = 2013 -  Train['Outlet_Establishment_Year']
Train = Train.drop('Outlet_Establishment_Year', axis=1)


# In[24]:


#Boxplots of features vs target

fig, axes = plt.subplots( nrows = 3, ncols = 2, figsize = (15,15))
fig.subplots_adjust(hspace = .3, wspace=.2)

for ax, i in zip(axes.ravel(), list(list(Train.columns)[i] for i in [2,4,6,7,8,9])):
    sns.boxplot(x=i, y='Item_Outlet_Sales', data=Train, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


# In[25]:


Item_Type_Outlet = Train.groupby(['Outlet_Identifier','Item_Type'])['Item_Outlet_Sales'].sum().round(2)
Item_Type_Outlet.unstack().plot(kind='bar', stacked=True, figsize=(8,6))
plt.ticklabel_format(axis="y", style='plain')
ax = plt.gca()
fmt = "{x:,.0f}"
tick = mpl.ticker.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.show()


# In[26]:


Outlet_Type_Size = Train.groupby(['Outlet_Type','Outlet_Size'])['Item_Outlet_Sales'].sum().round(2)
Outlet_Type_Size.unstack().plot(kind='bar', stacked=True, figsize=(8,6))
plt.ticklabel_format(axis="y", style='plain')
ax = plt.gca()
fmt = "{x:,.0f}"
tick = mpl.ticker.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.show()


# In[27]:


Outlet_Type_Location = Train.groupby(['Outlet_Type','Outlet_Location_Type'])['Item_Outlet_Sales'].sum().round(2)
Outlet_Type_Location.unstack().plot(kind='bar', stacked=True, figsize=(8,6))
plt.ticklabel_format(axis="y", style='plain')
ax = plt.gca()
fmt = "{x:,.0f}"
tick = mpl.ticker.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
plt.show()


# In[28]:


Train.groupby(['Outlet_Age', 'Outlet_Identifier'])['Outlet_Identifier'].nunique()


# In[29]:


Train['Item_MRP']=np.select([Train['Item_MRP']<75, Train['Item_MRP'].between(75,150),
                                     Train['Item_MRP'].between(150, 200)],['Low','Medium', 'High'],'Very High')

encoder= ce.OrdinalEncoder(cols=['Outlet_Size', 'Outlet_Location_Type', 'Item_Fat_Content', 'Outlet_Type',
                                'Item_MRP'], return_df=True,
                           mapping=[{'col':'Outlet_Size', 'mapping':{'Small':0,'Medium':1,'Large':2}},
                                    {'col':'Outlet_Location_Type', 'mapping' :{'Tier 1':0, 'Tier 2':1, 'Tier 3':2}},
                                    {'col':'Item_Fat_Content', 'mapping' :{'Non Food':0, 'Low Fat':1, 'Regular':2}},
                                    {'col':'Outlet_Type', 'mapping' :{'Grocery Store':0, 'Supermarket Type1':1,
                                                                      'Supermarket Type2':2, 'Supermarket Type3':3}},
                                    {'col': 'Item_MRP', 'mapping' :{'Low':0, 'Medium':1, 'High':2, 'Very High':3}}])



Train = encoder.fit_transform(Train)


# In[30]:


encoder=ce.OneHotEncoder(cols='Item_Type',handle_unknown='return_nan',return_df=True,use_cat_names=True)
Train = encoder.fit_transform(Train)

encoder = ce.count.CountEncoder(cols=['Item_Identifier'])
Train = encoder.fit_transform(Train)

encoder= ce.BaseNEncoder(cols=['Outlet_Identifier'],base=5, drop_invariant=True)
Train = encoder.fit_transform(Train)


# In[31]:



corrMatrix = (Train.corr(method= 'spearman'))

fig, ax = plt.subplots(figsize=(15,10))

ax = sns.heatmap(corrMatrix, vmin=-1, vmax=1, annot=True, fmt='.2f', center=0, cmap=sns.diverging_palette(20, 220, n=200),
                 linewidths=0.5)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()



# <h1>Linear Regression</h1>

# In[32]:


X = Train.drop('Item_Outlet_Sales', axis=1)
Y = Train['Item_Outlet_Sales']


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2021)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_predicted = model.predict(X_test)


# In[34]:


LR_MSE = mean_squared_error(Y_test, Y_predicted, squared = False)
LR_Score = model.score(X_test, Y_test)


# In[35]:


coef = pd.Series(model.coef_,X_train.columns).sort_values()

coef.plot(kind='bar', title='Modal Coefficients')
plt.show()


# <h1>Elastic Net Regression</h1>

# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2021)

model = ElasticNet(alpha=0.01, l1_ratio=0.1, normalize=False)

model.fit(X_train,Y_train)

Y_predicted = model.predict(X_test)


# In[37]:


ENR_MSE = mean_squared_error(Y_test, Y_predicted, squared = False)
ENR_Score = model.score(X_test, Y_test)


# In[38]:


coef = pd.Series(model.coef_,X_train.columns).sort_values()

coef.plot(kind='bar', title='Modal Coefficients')
plt.show()


# Both linear regression and elastic net give larger coefficients to the same features.

# <h1>KNeighbours Regression</h1>

# In[39]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2021)

norm = MinMaxScaler(feature_range=(0, 1))

normX = norm.fit(X_train)
X_train = normX.transform(X_train)
X_test = normX.transform(X_test)


# In[40]:


params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

knn = KNeighborsRegressor()

model_param = GridSearchCV(knn, params, cv=5)
model_param.fit(X_train,Y_train)


# In[41]:


rmse_val = []
for K in range(20):
    K = K+1
    model = KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, Y_train)  #fit the model
    Y_predicted = model.predict(X_test) #make prediction on test set
    error = mean_squared_error(Y_test, Y_predicted, squared = False) #calculate rmse
    rmse_val.append(error) #store rmse values


curve = pd.DataFrame(rmse_val, columns=['RMSE_curve']) #elbow curve 

ax = curve.plot(xticks=[int(x) for x in range(20)])
ax.set_xlabel('Neighbours')
ax.set_ylabel('RMSE')
plt.show()


# In[42]:


model = KNeighborsRegressor(n_neighbors = model_param.best_params_['n_neighbors'])

model.fit(X_train, Y_train)
Y_predicted = model.predict(X_test)
KNN_MSE = mean_squared_error(Y_test, Y_predicted, squared = False)
KNN_Score = model.score(X_test, Y_test)


# <h1>SVR</h1>

# In[43]:


params = {'kernel':['rbf', 'poly', 'linear', 'sigmoid']}

model = SVR()

model_svr = GridSearchCV(model, params, cv=5)
model_svr.fit(X_train,Y_train)


# In[44]:


model = SVR(kernel=model_svr.best_params_['kernel'])

model.fit(X_train,Y_train)

Y_predicted = model.predict(X_test)
SVR_MSE = mean_squared_error(Y_test, Y_predicted, squared = False)
SVR_Score = model.score(X_test, Y_test)


# <h1>Decision Tree Regressor</h1>

# In[45]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2021)

dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

dtree.fit(X_train, Y_train)
Y_predicted = dtree.predict(X_test)
DTR_MSE = mean_squared_error(Y_test, Y_predicted, squared = False)
DTR_Score = dtree.score(X_test, Y_test)


# <h1>Random Forest</h1>

# In[46]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2021)

model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)

model_rf.fit(X_train, Y_train) 
Y_predicted = model_rf.predict(X_test)
RF_MSE = mean_squared_error(Y_test, Y_predicted, squared = False)
RF_Score = model_rf.score(X_test, Y_test)


# <h1>Comparison</h1>

# In[47]:


Model_Performance = pd.DataFrame({'Model': ['Linear', 'Elastic Net', 'KNN', 'SVR', 'Decision Tree', 'Random Forest'],
                                 'RMSE' : [LR_MSE, ENR_MSE, KNN_MSE, SVR_MSE, DTR_MSE, RF_MSE],
                                 'R2 Score' : [LR_Score, ENR_Score, KNN_Score, SVR_Score, DTR_Score, RF_Score]})
Model_Performance.round(2).sort_values(by='R2 Score', ascending=False)


# The best model for this dataset is Random forest, with RMSE of 326.13 and 96% R^2 score.
