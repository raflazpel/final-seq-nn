# Import needed packages
import numpy as np
import pandas as pd
import random
from collections import deque
from sklearn import preprocessing
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor

class BitcoinPredictorPipeline:
    """ This class represents a whole criptocurrency price predictor. Parameters like currency,
     length of the sequence used or number of epochs in training can be fine-tuned"""

    def __init__(self, past_length=40, future_distance=2):
        self.past_length = past_length
        self.future_distance = future_distance
        self.data = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.model = None

    def download_data(self, criptocoin='BTC', currency='EUR'):
        # Alpha Vantage API
        api_key = "J41D51QYZT0JLZ99"
        # This API outputs a dataframe with the open, low, high and close values in the "market" currency
        # daily for the whole historical record. Also gives same values for USD and the volume and market capitalization
        # in USD.
        cc = CryptoCurrencies(key=api_key, output_format='pandas')
        data, _ = cc.get_digital_currency_daily(symbol=criptocoin, market=currency)

        data = data.rename(columns={"3a. low (EUR)": "low",
                                    "2a. high (EUR)": "high",
                                    "1a. open (EUR)": "open",
                                    "4a. close (EUR)": "close",
                                    "5. volume": "volume" })

        # Keep only the "close" and "volume" columns
        self.data = data[["close", "volume"]]

    def separate_train_test(self):

        # Fill possible NaN's. (Non-existant)
        self.data.fillna(method="ffill", inplace=True)
        # Create target variable by shifting the closing value as many days as "future_distance"
        self.data['future'] = self.data['close'].shift(-self.future_distance)
        self.data.dropna(inplace=True)

        # Sort values
        times = sorted(self.data.index.values)

        # Separate last 10% for test values and rest for training
        last_records = sorted(self.data.index.values)[-int(0.1 * len(times))]
        test_df = self.data[(self.data.index >= last_records)]
        train_df = self.data[(self.data.index < last_records)]

        self.train_x, self.train_y = self.preprocess_df(train_df)
        self.test_x, self.test_y = self.preprocess_df(test_df)

    def preprocess_df(self, df):

        # Scale values between 0 and 10.
        df = self.__scale(df)

        # This list will contain all the sequences
        sequential_data = []
        # This deque structure will hold each sequence
        prev_days = deque(maxlen=self.past_length)

        for i in df.values:  # iterate over the rows
            # Append close and volume columns
            prev_days.append([n for n in i[:-1]])
            # If we recorded all the values of the sequence (self.past_length) we add it to sequential_data
            if len(prev_days) == self.past_length:
                # Each element of sequential_data contains all independent columns for the last "past_lenght" elements
                # and the target value.
                sequential_data.append([np.array(prev_days), i[-1]])

        # As each element of "sequential_data" can now be independently used for train or test, we can (must) shuffle it
        random.shuffle(sequential_data)
        x = []
        y = []
        for seq, target in sequential_data:
            x.append(seq)
            y.append(target)
        # Return the sequences and their target values
        return np.array(x), y  # return X and y...and make X a numpy array!

    def __scale(self, df):
        # go through all of the columns
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
        scaler.fit(df.values)
        df = scaler.transform(df.values)
        return pd.DataFrame(df)

    def build_model(self):
        # Let's build the model:
        model = Sequential()
        model.add(LSTM(128, input_shape=(pipeline.train_x.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))
        # Normalize activation outputs:
        model.add(BatchNormalization())

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(15, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='linear'))
        # Model compile settings:
        opt = Adam(lr=0.001, decay=1e-6)

        # Compile model
        model.compile(
            loss='mse',
            optimizer=opt
        )

        model.summary()
        self.model = model

    def train_model(self, batch_size=64, epochs=5):
        history_fitting = self.model.fit(
            self.train_x, self.train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.test_x, self.test_y))
        return history_fitting

    def evaluate_visualize_model(self, history_records):
        # Calculate and print the mse
        score = self.model.evaluate(self.test_x, self.test_y, verbose=1)
        print('Test mse:', score)

        # Baseline model
        baseline_median = np.median(self.train_y)
        from sklearn.metrics import mean_squared_error
        mse_baseline = mean_squared_error(np.asarray(self.test_y), np.full(shape=len(self.test_y), fill_value=baseline_median, dtype=np.float))

        # Loss per epoch
        plt.plot(history_records.history['loss'])
        plt.plot(history_records.history['val_loss'])
        plt.axhline(y=mse_baseline, color='r', linestyle='-')
        plt.title('Loss per epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

# Instantiate the class
pipeline = BitcoinPredictorPipeline(30, 1)

# Download the data
pipeline.download_data()
pipeline.separate_train_test()
print("Training records: %s" % len(pipeline.train_x))
print("Validation records: %s" % len(pipeline.test_x))
pipeline.build_model()
history = pipeline.train_model()
pipeline.evaluate_visualize_model(history)

