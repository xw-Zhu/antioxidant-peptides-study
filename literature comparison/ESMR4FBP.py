

## import libraries
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score  # 模型评估指标
warnings.filterwarnings("ignore")  # 忽略告警

# Load the data
X_new = pd.read_csv('C:/Users/li/Desktop/416465/encoded_data.csv', delimiter=',',header = None,index_col= False, skiprows=0)
X = X_new.values
print(X.shape)
y = pd.read_csv('C:/Users/li/Desktop/ESMR4FBP_ZJU-main/data/antioxidant_tripepetide_data/y_data_ABTS.csv',header=None).values
print(y.shape)

# ## LSTM model development

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#LSTM
from keras.optimizers import Adam

# dataset processing
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])
input_shape = (1, X_train.shape[2])

# bulid RNN model
model = Sequential()

model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(50, activation='tanh'))
model.add(Dropout(0.15))
model.add(Dense(25, activation='tanh'))
model.add(Dropout(0.15))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate = 0.001), loss='mean_squared_error')
# 训练模型
history = model.fit(X_train, y_train, epochs=96, batch_size=16, verbose=1,validation_split=0.2)#,callbacks=[early_stopping])

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
y_pred_train = y_pred_train.reshape(y_pred_train.shape[0], y_pred_train.shape[1])

# Calculate the MSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Test RMSE: {rmse}')

r2 = r2_score(y_test, y_pred)
print(f'Test R^2: {r2}')
r2_train = r2_score(y_train, y_pred_train)
print(f' R^2_train: {r2_train}')
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
print(f' MSE_train: {rmse_train}')
print('EVS_test:', round(explained_variance_score(y_test, y_pred), 4))
print('AE_test:', round(mean_absolute_error(y_test, y_pred), 4))

