#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import math
from datetime import datetime

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf


# In[2]:


##importing data file
## Importing data file
df = pd.read_csv('Crack_eia.csv',index_col='Date', parse_dates=True)
df=df.dropna()
print('Shape of data',df.shape)
df.head()


# In[3]:


print (df.info())


# In[4]:


# create new variables
df[ 'lcrack'] = np.log(df['crack'])
df['lwti' ] = np.log(df['wti'])
df['lgas' ] = np.log(df['gasoline'])
df['lheat' ] = np.log(df['heatoil'])


# In[5]:


#
# non-stationary - use first difference
df['dcrack'] = df.crack.diff(1)
df['dlcrack'] = df.lcrack.diff(1)
df['dwti'] = df.wti.diff(1)
df['dlwti'] = df.lwti.diff(1)


# In[6]:


# create lagged variables for crack spread
df['lcrack_L1'] = df['lcrack'].shift(1)
df['lcrack_L2'] = df['lcrack'].shift(2)
df['lcrack_L3'] = df['lcrack'].shift(3)
df['lcrack_L4'] = df['lcrack'].shift(4)


# In[7]:


# create lagged variables for WTI crude oil
df['lwti_L1'] = df['lwti'].shift(1)
df['lwti_L2'] = df['lwti'].shift(2)
df['lwti_L3'] = df['lwti'].shift(3)
df['lwti_L4'] = df['lwti'].shift(4)


# In[8]:


# Regression brent oil, crack spread and it's lag variables
mod_mlr2 = smf.ols('lwti ~ lgas + lheat + lcrack + lcrack_L1 + lcrack_L2 + lcrack_L3 + lcrack_L4',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=4)
print(mod_mlr2.summary())
print()


# In[9]:


from statsmodels.stats.stattools import durbin_watson

## perform Durbin-Watson test for serial correlation
durbin_watson(mod_mlr2.resid)


# In[10]:


# Regression crack spread and it's lag variables
mod_mlr3 = smf.ols('lcrack ~ dlcrack + lcrack_L1 + lcrack_L2 + lcrack_L3 + lcrack_L4',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=4)
print(mod_mlr3.summary())
print()


# In[11]:


## Importing arima packages        
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


# In[12]:


df=df.dropna()
##Arima model
stepwise_fit = auto_arima(df['lwti'], trace=True, 
    suppress_warnings=True)
stepwise_fit.summary()


# In[13]:


## importing ARIMA Functions
from statsmodels.tsa.arima.model import ARIMA


# In[14]:


## Spliting data set into training and testing
print (df.shape)
train=df.iloc[:-1900]
test=df.iloc[-1900:]
print(train.shape,test.shape)


# In[35]:


## Train the model
model1=ARIMA(train['lwti'],order=(1,1,1))
model1=model1.fit()
model1.summary()


# In[36]:


## Make predictions on test set
start=len(train)
end=len(train)+len(test)-1
pred=model1.predict(start=start,end=end,typ='levels')
print(pred)
pred.index=df.index[start:end+1]
print(pred)


# In[37]:


## Plotting the prediction
pred.plot(legend=True,figsize=(14,8))
test['lwti'].plot(legend=True)

plt.axhline(0, linestyle='--', color='r', alpha=0.6)


# In[38]:


## Testing if it's a good model fitted
test['lwti'].mean()


# In[39]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,test['lwti']))
print(rmse)


# In[40]:


mae=mean_absolute_error(pred,test['lwti'])
print(mae)


# In[41]:


mse=mean_squared_error(pred,test['lwti'])
print(mse)


# In[42]:


rmse2=np.sqrt(mse)
print(rmse2)


# In[43]:


aic = model1.aic
print(aic)


# In[44]:


# Print results
print('In-Sample Measures:')
print('MSE: %.3f' % mse)
print('RMSE: %.3f' % rmse2)
print('MAE: %.3f' % mae)
print('AIC: %.3f' % aic)


# In[45]:


model1.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[46]:


df=df.dropna()
##Arima model
stepwise_fit2 = auto_arima(df['lcrack'], trace=True, 
    suppress_warnings=True)
stepwise_fit2.summary()


# In[27]:


## Spliting data set into training and testing
print (df.shape)
train=df.iloc[:-1900]
test=df.iloc[-1900:]
print(train.shape,test.shape)


# In[47]:


## Train the model
model2=ARIMA(train['lcrack'],order=(4,1,5))
model2=model2.fit()
model2.summary()


# In[48]:


# Make predictions on test set
start=len(train)
end=len(train)+len(test)-1
pred2=model2.predict(start=start,end=end,typ='levels')
print(pred2)
pred2.index=df.index[start:end+1]
print(pred2)


# In[49]:


## Plotting the prediction
pred2.plot(legend=True,figsize=(14,8))
test['lcrack'].plot(legend=True)

plt.axhline(0, linestyle='--', color='r', alpha=0.6)


# In[52]:


## Testing if it's a good model fitted
test['lcrack'].mean()


# In[53]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,test['lcrack']))
print(rmse)


# In[54]:


mse=mean_squared_error(pred,test['lcrack'])
rmse=sqrt(mean_squared_error(pred,test['lcrack']))
mae=mean_absolute_error(pred,test['lcrack'])

# Print results
print('In-Sample Measures:')
print('MSE: %.3f' % mse)
print('RMSE: %.3f' % rmse)
print('MAE: %.3f' % mae)


# In[55]:


model2.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[ ]:




