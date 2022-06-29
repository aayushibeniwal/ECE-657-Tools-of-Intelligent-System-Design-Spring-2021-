# import required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.losses import MeanSquaredError as MSE
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from pickle import dump, load
from sklearn.metrics import mean_squared_error as MSE

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
    # load the scaler from train py
    minMax_data = load(open('data/minMax_data.pkl', 'rb'))
    minMax_label = load(open('data/minMax_label.pkl', 'rb'))

    print("LOADING saved RNN model....")
	# 1. Load your saved model
    RNN_model = load_model('models/Group_44_RNN_model.h5')

    print("LOADING test_data_RNN.csv...Please Wait...")
	# 2. Load your testing data
    test_dataset = pd.read_csv("data/test_data_RNN.csv")

    # Preprocess and prepare the data to be used with the RNN model

    # Scale Test Data
    x_test = minMax_data.fit_transform(test_dataset.iloc[:, :-1])
    # Scale Test labels
    y_test = minMax_label.fit_transform(test_dataset.iloc[:, -1:])

    x_test = np.array(x_test)
    x_test_3D = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print("RUNNING predictions for Test data.......")
	# 3. Run prediction on the test data and output required plot and loss
    testing_loss = RNN_model.evaluate(x_test_3D, y_test)
    print("\nTesting Loss is evaluated and is approximately --> ", testing_loss)
    print("--------------------------------------------------------------------------")

    # Predict the Opening price for test dataset
    test_predict = RNN_model.predict(x_test_3D)
    test_predict_unscaled = minMax_label.inverse_transform(test_predict)
    actual_test_label = minMax_label.inverse_transform(y_test)

    loss_test = MSE(actual_test_label, test_predict_unscaled)
    print("--------------------------------------------------------")
    print("| The mean squared error of TESTING is: {}  |".format(loss_test))
    print("--------------------------------------------------------")


    print("\n\nVISUALIZE Actual Price VS Predicted Opening Price for Test data")
    # Visualize Actual Price VS Predicted Price for Test data
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.title("Test data plot for Actual vs Predicted Opening price")
    plt.plot(actual_test_label, color='red', label='Actual Opening Price')
    ax.plot(test_predict_unscaled, color='blue', label='Predicted Opening Price')
    plt.ylabel('Price in $')
    plt.xlabel('Days')
    plt.legend()
    print("VISUALIZING please wait.....")
    plt.show()