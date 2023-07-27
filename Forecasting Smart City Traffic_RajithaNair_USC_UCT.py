#!/usr/bin/env python
# coding: utf-8

# ###### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load train and test datasets
train_data = pd.read_csv("C:/Users/RAJITHA/OneDrive/Desktop/Upskill/DS & ML/Project9_smart-city-traffic-patterns/archive (6)/train_aWnotuB.csv")
test_data = pd.read_csv("C:/Users/RAJITHA/OneDrive/Desktop/Upskill/DS & ML/Project9_smart-city-traffic-patterns/archive (6)/test_BdBKkAj.csv")

print(train_data.info())
print(train_data.head())


# ###### Convert datetime to separate columns for year, month, day, hour.

# In[3]:


train_data['DateTime'] = pd.to_datetime(train_data['DateTime'])
train_data['year'] = train_data['DateTime'].dt.year
train_data['month'] = train_data['DateTime'].dt.month
train_data['day'] = train_data['DateTime'].dt.day
train_data['hour'] = train_data['DateTime'].dt.hour

test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])
test_data['year'] = test_data['DateTime'].dt.year
test_data['month'] = test_data['DateTime'].dt.month
test_data['day'] = test_data['DateTime'].dt.day
test_data['hour'] = test_data['DateTime'].dt.hour

# Drop the original datetime column
train_data = train_data.drop(columns=['DateTime'])
test_data = test_data.drop(columns=['DateTime'])


print(train_data.head())
print(train_data.info())


# ###### Visualizations

# In[4]:


# Distribution of the target variable 'vehicles' in the training data
plt.figure(figsize=(8, 6))
sns.histplot(train_data['Vehicles'], kde=True, color='skyblue')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.title('Distribution of Vehicles in the Training Data')
plt.show()


# In[5]:


# Scatter plot of 'vehicles' against 'hour'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='hour', y='Vehicles', data=train_data, alpha=0.5)
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Vehicles')
plt.title('Number of Vehicles vs. Hour of the Day')
plt.show()

# Box plot of 'vehicles' against 'month'
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='Vehicles', data=train_data)
plt.xlabel('Month')
plt.ylabel('Number of Vehicles')
plt.title('Number of Vehicles vs. Month')
plt.show()


# ###### Separating the target variable 'vehicles' and spliting the data into training and validation sets

# In[6]:


X = train_data.drop(columns=['Vehicles'])
y = train_data['Vehicles']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


X_train


# #### XGBOOST MODEL

# In[8]:


import xgboost as xgb
# Initialize the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)


# ###### Mean squared error on the validation set

# In[9]:


mse = mean_squared_error(y_val, y_pred)
print("Mean Squared Error:", mse)


# ###### Making predictions on the test set

# In[10]:


test_predictions = model.predict(test_data)

# Adding the predictions to the test data
test_data['predicted_vehicles'] = test_predictions

# Converting the separate columns for year, month, day, hour back to a datetime column
test_data['datetime'] = pd.to_datetime(test_data[['year', 'month', 'day', 'hour']])
test_data = test_data.drop(columns=['year', 'month', 'day', 'hour'])

# Saving the predictions to a CSV file
test_data.to_csv("traffic_predictions.csv", index=False)


# ###### Predictions

# In[11]:


predictions_df = pd.read_csv("traffic_predictions.csv")
print(predictions_df.head())


# In[ ]:





# #### Random Forest Regressor Model

# ###### Initializing and training the Random Forest Regressor model

# In[15]:


model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_rf = model_rf.predict(X_val)


# ###### Mean squared error on the validation set

# In[16]:


mse_rf = mean_squared_error(y_val, y_pred_rf)
print("Random Forest Mean Squared Error:", mse_rf)


# ###### Making predictions

# In[17]:


test_predictions_rf = model_rf.predict(test_data)

# Add the predictions to the test data
test_data['predicted_vehicles_rf'] = test_predictions_rf

test_data['datetime'] = pd.to_datetime(test_data[['year', 'month', 'day', 'hour']])
test_data = test_data.drop(columns=['year', 'month', 'day', 'hour'])


# In[18]:


# Save the predictions to a CSV file
test_data.to_csv("traffic_predictions_rf.csv", index=False)


# In[19]:


# Read the predictions CSV file into a DataFrame
predictions_df = pd.read_csv("traffic_predictions_rf.csv")

# Display the contents of the DataFrame
print(predictions_df.head())


# In[ ]:





# #### SVR

# ###### Scale the features using StandardScaler

# In[21]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_data_scaled = scaler.transform(test_data)


# In[22]:


# Initialize the SVR model
model_svr = SVR(kernel='rbf')

# Train the model on the scaled training data
model_svr.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred_svr = model_svr.predict(X_val_scaled)


# ###### Mean squared error on the validation set

# In[23]:


mse_svr = mean_squared_error(y_val, y_pred_svr)
print("SVR Mean Squared Error:", mse_svr)


# ###### Make predictions on the scaled test set

# In[24]:


test_predictions_svr = model_svr.predict(test_data_scaled)

# Add the predictions to the test data
test_data['predicted_vehicles_svr'] = test_predictions_svr

# If needed, convert the separate columns for year, month, day, hour back to a datetime column
test_data['datetime'] = pd.to_datetime(test_data[['year', 'month', 'day', 'hour']])
test_data = test_data.drop(columns=['year', 'month', 'day', 'hour'])

# Save the predictions to a CSV file
test_data.to_csv("traffic_predictions_svr.csv", index=False)


# ###### Predictions

# In[25]:


predictions_svr_df = pd.read_csv("traffic_predictions_svr.csv")

# Display the contents of the DataFrame
print(predictions_svr_df.head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




