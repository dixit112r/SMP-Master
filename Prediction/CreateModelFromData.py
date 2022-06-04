import sqlite3 as sq
import pandas as pd
import io
import os

from SMP.settings import BASE_DIR
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


def CreateModelFromData(ticker, table):
    csv_path = os.path.join(BASE_DIR, "static_files/datasets/" + ticker + ".csv")
    model_path = os.path.join(BASE_DIR, "static_files/datasets/" + ticker + ".h5")

    sql_data = BASE_DIR / 'db.sqlite3'
    conn = sq.connect(sql_data)
    data = pd.read_sql_query("SELECT * FROM " + table, conn)
    data.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path, parse_dates=True)
    dates = df['Date'][:int(len(df)*.8401)]
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    df = df.dropna()
    # ?--------------------------------------------------------------------------------
    print(df)
    # ?--------------------------------------------------------------------------------

    # CONST VALUE
    steps = 150

    # os.remove(csv_path)

    df1=df.reset_index()['Close']
    # ?--------------------------------------------------------------------------------
    # print(df1)
    # ?--------------------------------------------------------------------------------

    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    # ?--------------------------------------------------------------------------------
    # print(df1.shape)
    # ?--------------------------------------------------------------------------------

    training_size=int(len(df1)*0.75)
    test_size=int(len(df1)*.15)
    dataset_length = training_size+test_size
    train_data,test_data=df1[0:training_size],df1[training_size:dataset_length]
    # ?--------------------------------------------------------------------------------
    # print(training_size, test_size) 
    # ?--------------------------------------------------------------------------------

    time_step = 150
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)
    # ?--------------------------------------------------------------------------------
    # print(x_train.shape), print(y_train.shape)
    # print(x_test.shape), print(y_test.shape)
    # ?--------------------------------------------------------------------------------

    # reshape input to be [samples, time steps, features] which is required for LSTM
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Create the Stacked LSTM model
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(steps,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    # ?--------------------------------------------------------------------------------
    print(model.summary())
    # ?--------------------------------------------------------------------------------

    # Lets Do the prediction and check performance metrics
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    model.fit(x_train,y_train,validation_data = (x_test,y_test), epochs = 100, batch_size = 64,verbose = 1)

    model.save(model_path)