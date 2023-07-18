#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from itertools import cycle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle


# In[2]:


#Loading data
df = pd.read_excel('Crude_oil_WTI.xls')
df.set_index('Date')
df.head()


# In[3]:


df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.week


# In[4]:


df


# In[5]:


df = df.set_index(['Date'])


# In[6]:


df_new = df.iloc[:,[0]]
df_new.head()


# In[7]:


# Create an instance of the scaler
scaler = MinMaxScaler()

# Fit the scaler to your data
scaler.fit(df_new)

# Transform your data using the scaler
df_new = scaler.transform(df_new)


# In[8]:


df_new.shape


# In[9]:


# 0.80 - 0.20 train and test split dividing input class and target class 
training_size=int(len(df_new)*0.80)
test_size=len(df_new)-training_size
train_data,test_data=df_new[0:training_size,:],df_new[training_size:len(df_new),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)


# In[10]:


# convert an array of values into a dataset matrix

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[11]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 150
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)


# In[12]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)


# In[13]:


tf.keras.backend.clear_session()
model=Sequential()
model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[14]:


model.summary()


# In[15]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=1)


# In[16]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict.shape, test_predict.shape


# In[17]:


# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))


# In[18]:


import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
# Evaluation metrices RMSE and MAE
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
print("Test data MAE: ", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))


# In[19]:


print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict)*100)
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict)*100)


# In[20]:


train_r2_LSTM=r2_score(original_ytrain, train_predict)
train_r2_LSTM=r2_score(original_ytest, test_predict)
print("Train data R2 score:", train_r2_LSTM)
print("Test data R2 score:", train_r2_LSTM)


# In[21]:


df.head()


# In[22]:


df = df.reset_index()
df['Date']


# In[23]:


# shift train predictions for plotting

look_back=time_step
trainPredictPlot = np.empty_like(df_new)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(df_new)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df_new)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original Oil Price','Train Predicted Oil Price','Test Predicted Oil Price'])


plotdf = pd.DataFrame({'Date': df['Date'],
                       'Original Oil Price': df['Price'],
                      'Train Predicted Price': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'Test Predicted Price': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['Date'], y=[plotdf['Original Oil Price'],plotdf['Train Predicted Price'],
                                          plotdf['Test Predicted Price']],
              labels={'value':'Price','Date': 'Date'})
fig.update_layout(title_text='Comparision between Original Oil Price vs Predicted Oil Price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[24]:


original_ytest=  pd.DataFrame(original_ytest, columns =['Actual',])


# In[25]:


test_predict=pd.DataFrame(test_predict, columns =['Predicted',])


# In[26]:


original_ytest['predicted']=test_predict
comparison=original_ytest
comparison


# In[27]:


comparison.plot(figsize=(15,8))

plt.show()


# In[29]:


pickle.dump(model, open('lstm_model.pickle','wb'))


# In[ ]:




