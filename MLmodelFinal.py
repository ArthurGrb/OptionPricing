import numpy as np
import pricers as p
import optionValuation as ov
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from datetime import datetime
import pandas as pd
from matplotlib import pyplot
import matplotlib as mpl
from keras.models import load_model
import time
import random

def load(model):
    # load json and create model
    json_file = open("Models/" + str(model) + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Models/" + str(model) + ".h5")
    print("Loaded model from disk")
    return loaded_model

def save(model, str1):
    model_json = model.to_json()
    with open(str1 + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(str1 + ".h5")
    print("Saved model to disk")

options = pd.read_pickle('C:/Users/phili/PycharmProjects/AmericanOptionPricer-ML/options-dataset-v1.pkl')
options = options.sort_values('BinomialTree', ascending=False)
options = options.drop(options[(options['BinomialTree'] / options['Bjerksund-Stensland'] > 4.0) & (options['BinomialTree'] > 1.0)].index)

mse = mean_squared_error(options['BinomialTree'], options['Black-Scholes'])

x = options[['S', 'K', 'T', 'r', 'sigma', 'Black-Scholes', 'Bjerksund-Stensland']]
# x = options[['S', 'K', 'T', 'r', 'sigma']]
y = options['BinomialTree']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2)

now = datetime.now()
checkpoint_path = "Models/model_" + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(
    now.hour) + "_" + str(now.minute) + ".h5"

def nn(L_SIZE, LAYERS, DROPOUT, ACT_F):
    start = time.time()
    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=50, mode='min'),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    ]

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(xtrain.shape[1],)))
    for i in range(1,LAYERS):
        model.add(keras.layers.Dense(L_SIZE, activation=ACT_F))
        model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer='Adam', loss='mse')

    history = model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=100000, batch_size=2**18,
                        callbacks=keras_callbacks, verbose=2)

    end = time.time()
    print("Training time: ", end - start)
    # plot training history
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

    # model = load("model_2019_11_5_20_3")
    yhat = model.predict(xtest)
    df = xtest.reset_index(drop=True)
    df['ytest'] = ytest.reset_index(drop=True)
    df['yhat'] = yhat
    df['yhat'] = np.maximum(df['yhat'], df['Black-Scholes'])

    mse_nn = mean_squared_error(df['ytest'], df['yhat'])

    save(model, "Models/allInputs-" + str(L_SIZE) + "-" + str(LAYERS) + "-" + str(DROPOUT) + "-relu")

    return L_SIZE, LAYERS, DROPOUT, ACT_F, model, mse_nn


L_SIZE_LIST = [64, 128]
LAYERS_LIST = list(range(3, 11))
DROPOUT_LIST = [0.0, 0.2]
ACT_F_LIST = [tf.nn.relu]

best_L_SIZE = 0
best_LAYERS = 0
best_DROPOUT = 0.0
best_ACT_F = None
best_model = None
best_mse = 1000000000000.0

while True:
    L_SIZE = random.choice(L_SIZE_LIST)
    LAYERS = random.choice(LAYERS_LIST)
    DROPOUT = random.choice(DROPOUT_LIST)
    ACT_F = random.choice(ACT_F_LIST)
    print(L_SIZE, LAYERS, DROPOUT, ACT_F)
    L_SIZE, LAYERS, DROPOUT, ACT_F, model, mse = nn(L_SIZE, LAYERS, DROPOUT, ACT_F)
    print("*****************************************")
    print("L_SIZE:", L_SIZE)
    print("LAYERS:", LAYERS)
    print("DROPOUT:", DROPOUT)
    print("ACT_F:", ACT_F)
    print("MSE:", mse)
    print("BEST_MSE:", best_mse)
    print("*****************************************")
    str1 = str(L_SIZE) + ',' + str(LAYERS) + ',' + str(DROPOUT) + ',' + str(ACT_F) + ',' + str(mse) + '\n'
    f = open('results_nn.txt', 'a')
    f.write(str1)
    f.close()
    if mse < best_mse:
        # best_L_SIZE = L_SIZE
        # best_LAYERS = LAYERS
        # best_DROPOUT = DROPOUT
        # best_ACT_F = ACT_F
        best_model = model
        best_mse = mse
# # plot training history
# mpl.rcParams['agg.path.chunksize'] = 10000
# pyplot.plot(ytest, label='actual')
# pyplot.plot(yhat, label='predicted')
# pyplot.legend()
# pyplot.show()

# # Create linear regression object
# regr = linear_model.LinearRegression()
#
# # Train the model using the training sets
# regr.fit(xtrain, ytrain)
#
# # Make predictions using the testing set
# yhat = regr.predict(xtest)
# df = xtest.reset_index(drop=True)
# df['BinomialTree'] = ytest.reset_index(drop=True)
# df['LinearRegression'] = np.maximum(yhat, df['Black-Scholes'])
#
# mse_regr = mean_squared_error(df['BinomialTree'], df['LinearRegression'])
#
# plt.scatter(df['BinomialTree'], df['LinearRegression'])
# plt.xlabel("Binomial Tree")
# plt.ylabel("Linear Regression")
# plt.title("Linear Regression Fit")
# plt.savefig('regression.png', transparent=True)
#
#









