#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels plotly xgboost lughtgbm tensorflow keras nltk opencv-python


# In[2]:


# import basic Python libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# import machine learning packages
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split
#from sklearn.cross_validation import StratifiOf the new features, the majority don't have any correlations that would be useful (or not intuitive). One that sticks out as possibly important is the category_code feature is negatively correlated with cost, regular_price, and current_price. A set of new correlations of note are the correlations between date (like week and month) and seasonality with ratio and promo1. These correlations could be useful in creating a robust machine learning algorithm for predicting future sales.
#Create the features and target data along with splitting into training and testing data.edShuffleSplit

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

# import necessary seasonality decomposition and ARIMA modeling packages
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


# In[3]:


# import transactional sales data of articles into a pandas dataframe
sales_data = pd.read_csv("sales.txt", sep = ";", parse_dates=['retailweek'])

# import attribute data of sold articles
attribute_data = pd.read_csv("articles.txt", sep = ";")

data_joined = pd.merge(sales_data, attribute_data, how = 'inner', on = 'article')
data_joined.head()


# In[4]:


# create date columns
data_joined['week'] = data_joined['retailweek'].dt.isocalendar().week
data_joined['month'] = data_joined['retailweek'].dt.month
data_joined['year'] = data_joined['retailweek'].dt.year


# In[5]:


# create season column
data_joined['season'] = data_joined['retailweek'].apply(lambda dt: (dt.month%12 + 3)/3)


# In[6]:


# country
data_joined['country'] = pd.Categorical(data_joined['country'])
data_joined['country_code'] = data_joined.country.cat.codes


# In[7]:


# clothing category
data_joined['category'] = pd.Categorical(data_joined['category'])
data_joined['category_code'] = data_joined.category.cat.codes


# In[8]:


# product group of clothing
data_joined['productgroup'] = pd.Categorical(data_joined['productgroup'])
data_joined['productgroup_code'] = data_joined.productgroup.cat.codes


# In[9]:


# profit margin
data_joined['profit'] = data_joined['current_price'] - data_joined['cost']


# In[10]:


data_joined['total_profit'] = data_joined['profit']*data_joined['sales']
data_joined


# In[11]:


data_grouped = data_joined.groupby(['category', 'productgroup', 'article']).agg(['count'])
data_grouped.max()


# In[12]:


data_grouped.max


# In[13]:


data_EZ8648 = data_joined[data_joined['article']=='EZ8648']
data_EZ8648.head(100)


# In[14]:


data_EZ8648.shape


# In[15]:


data_EZ8648.describe()



# In[16]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (14,14))
data_EZ8648.plot(ax = axes[0,0], x = 'retailweek', y = 'sales')
axes[0,0].set_title('Article sales')

data_EZ8648.plot(ax = axes[0,1], y = 'sales', kind = 'hist')
axes[0,1].set_title('Article sales histogram')

data_EZ8648.plot(ax = axes[1,0], y = 'sales', kind = 'box')
axes[1,0].set_title('Article sales boxplot')

axes[1,1].set_title('INTENTIONALLY LEFT BLANK')
plt.show()


# In[17]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (14,14))
data_EZ8648.plot(ax = axes[0,0], x = 'retailweek', y = ['regular_price', 'current_price'])
axes[0,0].set_title('Article pricing')

data_EZ8648.plot(ax = axes[0,1], y = ['regular_price', 'current_price'], kind = 'hist')
axes[0,1].set_title('Article pricing histogram')

data_EZ8648.plot(ax = axes[1,0], y = ['regular_price', 'current_price'], kind = 'box')
axes[1,0].set_title('Article pricing boxplot')

data_EZ8648.plot(ax = axes[1,1], x = 'retailweek', y = ['regular_price', 'current_price'])
data_EZ8648.plot(ax = axes[1,1], secondary_y = True, x = 'retailweek', y = 'sales')
axes[1,1].set_title('Article sales and pricing')
plt.show()


# In[18]:


correlated_sales = data_EZ8648.select_dtypes(include = 'number').corr()
correlated_sales


# In[19]:


# plot the heatmap
sns.heatmap(correlated_sales)
plt.show()


# In[20]:


# define features I want to use and what the target data is
feature_list = ['regular_price', 'current_price', 'ratio', 'promo1', 'promo2', 'cost', 'week', 'month', 'year', \
               'season', 'country_code', 'category_code', 'productgroup_code', 'profit', 'total_profit']
target_list = ['sales']


# In[21]:


# subset joined dataframe into df_features and target
df_features = data_EZ8648[feature_list]
df_features.head()


# In[22]:


df_features.shape


# In[23]:


target = data_EZ8648[target_list]
target.head()


# In[24]:


target.shape


# In[25]:


feature_train, feature_test, target_train, target_test = train_test_split(df_features, target, test_size=0.3, random_state=42)


# In[26]:


print(feature_train.shape, feature_test.shape, target_train.shape, target_test.shape)


# In[27]:


print(type(feature_train), type(feature_test), type(target_train), type(target_train))


# In[28]:


#regressions
#LinReg = LinearRegression()
#Las = Lasso() # optimize with GridSearchCV: alpha=[0.1, 0.3, 0.5, 0.7, 0.9, 1] (default=1)
#DTree = DecisionTreeRegressor(random_state = 42)  # min_samples_leaf (default=1), min_samples_split (default=2)
RF = RandomForestRegressor(random_state = 42) # n_estimators (default=10), min_samples_leaf (default=1), min_samples_split (default=2)

# feature scaling
minMaxScaler = MinMaxScaler()
stdScaler = StandardScaler()

# dimensionality reduction and feature selection
PCAreducer = PCA(svd_solver = 'auto', random_state = 42)
KBestSelector = SelectKBest(f_regression) # optimize: k = [3, 5, 7, 10, 13, 15] (default = 10)


# In[30]:


# convert features and target to numpy arrays
target_np = target[target_list].values.ravel()
df_features_np = df_features[feature_list].values
target_train_np = target_train[target_list].values.ravel()
feature_train_np = feature_train[feature_list].values


# In[32]:


# create machine learning pipeline
pipeline = Pipeline([('scaler', minMaxScaler), ('PCA', PCAreducer), ('KBest', KBestSelector), ('reg', RF)])

# create parameters of GridSearchCV using pipeline
params = {
    'KBest__k': [3, 5, 8, 10, 15],
    'reg__n_estimators': [3, 5, 8, 10, 15],
    'reg__min_samples_split': [2, 4, 6, 8, 10, 12],
    'reg__min_samples_leaf': [1, 2, 4, 6, 8]
}

gridSearch = GridSearchCV(pipeline, param_grid = params, scoring = 'r2', verbose = 10) # scoring = 'r2',

# fit pipeline and grid search to training features and target dataset
gridSearch.fit(feature_train, target_train_np)

# store the best estimator (best classifier with parameters)
regr = gridSearch.best_estimator_


# In[34]:


# create handle
KBest_handle = gridSearch.best_estimator_.named_steps['KBest']

# get SelectKBest scores rounded to 2 decimal places
feature_scores = ['%.2f' % elem for elem in KBest_handle.scores_]

# create a tuple of SelectKBest feature names, scores and pvalues
feature_selected_tuple = [(feature_list[i], feature_scores[i]) \
                            for i in KBest_handle.get_support(indices=True)]

# sort by reverse score order
feature_selected_tuple = sorted(feature_selected_tuple, key = lambda feature: float(feature[1]), reverse=True)

# print selected feature names and scores
print(' ')
print('Selected Features, Scores:')
print(feature_selected_tuple)


# In[35]:


# print best parameters and corresponding score from gridSearchCV
print(' ')
print('Best parameters:')
print(gridSearch.best_params_)


print('')
print('Best score:')
print(gridSearch.best_score_)


# In[36]:


# make predictions based on test data split
pred = gridSearch.predict(feature_test)

# plot predictions (regression) vs target (truth)
x = np.arange(len(pred))
plt.figure()
plt.plot(x, target_test, '.')
plt.plot(x, pred, '-')
plt.show()


# In[37]:


# print results
print('mean_absolute_error:', mean_absolute_error(target_test, pred))
print('mean_squared_error:', mean_squared_error(target_test, pred))
print('median_absolute_error:', median_absolute_error(target_test, pred))
print('r2_score:', r2_score(target_test, pred))


# In[38]:


plt.figure()
data_EZ8648.plot( x = 'retailweek', y = 'sales')
plt.title('Article sales')
plt.show()


# In[40]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (14,14))
data_EZ8648.plot(ax = axes[0,0], x = 'retailweek', y = ['promo1', 'promo2'])
axes[0,0].set_title('Article promotions')

data_EZ8648.plot(ax = axes[0,1], y = ['promo1', 'promo2'], kind = 'hist')
axes[0,1].set_title('Article promotions histogram')

data_EZ8648.plot(ax = axes[1,0], y = ['promo1', 'promo2'], kind = 'box')
axes[1,0].set_title('Article promotions boxplot')

data_EZ8648.plot(ax = axes[1,1], x = 'retailweek', y = ['promo1', 'promo2'])
data_EZ8648.plot(ax = axes[1,1], secondary_y = True, x = 'retailweek', y = 'sales')
axes[1,1].set_title('Article sales and promotions')
plt.show()


# In[42]:


# Select only numeric columns
numeric_data = data_EZ8648.select_dtypes(include=['number'])

# Compute correlation matrix
correlated_articleSales = numeric_data.corr()

# Plot heatmap
sns.heatmap(correlated_articleSales)
plt.show()


# In[44]:


from statsmodels.tsa.seasonal import seasonal_decompose

data_EZ8648_dt = data_EZ8648.copy()

decompFreq = 1*3*4  # or just write 12

data_EZ8648_dt.reset_index(inplace=True)
data_EZ8648_dt = data_EZ8648_dt.set_index('retailweek')

decomp_articleSales = seasonal_decompose(data_EZ8648_dt.sales, model='additive', period=decompFreq)
decomp_articleSales.plot()
plt.show()


# In[45]:


data_EZ8648.groupby(by=['promo1'])['sales'].describe()


# In[46]:


data_EZ8648.groupby(by=['promo1'])['sales'].sum()


# In[47]:


data_EZ8648.groupby(by=['promo2'])['sales'].describe()


# In[48]:


data_EZ8648.groupby(by=['promo2'])['sales'].sum()


# In[49]:


data_EZ8648[(data_EZ8648.promo1 == 1) & (data_EZ8648.promo2 == 1)].shape


# In[50]:


g_p1 = data_EZ8648.groupby(by=['promo1'])
g_p1.apply(lambda x: x[x['promo2'] != 1]['sales'].describe())


# In[51]:


g_p1.apply(lambda x: x[x['promo2'] != 1]['sales'].sum())


# In[52]:


g_p2 = data_EZ8648.groupby(by=['promo2'])
g_p2.apply(lambda x: x[x['promo1'] != 1]['sales'].describe())


# In[ ]:




