# Import needed packages
import numpy as np
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

    def download_data(self, criptocoin='BTC', currency='USD'):
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
                                    "5. volume": "volume"})
        self.data = data

    def preprocess_basic(self):
        # Set time as index so we can join them on this shared time
        self.data.set_index("date", inplace=True)

        # Keep only the "close" and "volume" columns
        self.data = self.data[["close", "volume"]]

        # Fill NaN's
        self.data.fillna(method="ffill", inplace=True)
        # TODO why drop is necessary if we already filled nans?
        self.data.dropna(inplace=True)

        self.data['future'] = self.data['close'].shift(-self.future_distance)
        self.data.dropna(inplace=True)
        times = sorted(self.data.index.values)
        last_records = sorted(self.data.index.values)[-int(0.1 * len(times))]

        test_df = self.data[(self.data.index >= last_records)]
        train_df = self.data[(self.data.index < last_records)]

        self.train_x, self.train_y = self.preprocess_df(train_df)
        self.test_x, self.test_y = self.preprocess_df(test_df)

    def preprocess_df(self, df):

        df = self.__scale(df)

        sequential_data = []  # this is a list that will CONTAIN the sequences
        prev_days = deque(maxlen=self.past_length)

        for i in df.values:  # iterate over the values
            prev_days.append([n for n in i[:-1]])  # store all but the target
            if len(prev_days) == self.past_length:  # make sure we have 60 sequences!
                sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

        random.shuffle(sequential_data)  # shuffle for good measure.

        print(np.array(sequential_data).shape)

        random.shuffle(
            sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

        x = []
        y = []

        for seq, target in sequential_data:  # going over our new sequential data
            x.append(seq)  # X is the sequences
            y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

        print(x[0])
        print(y[0])
        return np.array(x), y  # return X and y...and make X a numpy array!

    def __scale(self, df):
        for col in df.columns:  # go through all of the columns
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.
        return df

    def build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(pipeline.train_x.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())  # normalizes activation outputs

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='linear'))
        # Model compile settings:

        opt = Adam(lr=0.001, decay=1e-6)

        # Compile model
        model.compile(
            loss='mse',
            optimizer=opt,
            metrics=['mape']
        )

        model.summary()
        self.model = model

    def train_model(self, batch_size=64, epochs=5):
        name = f"{self.past_length}-SEQ-{self.future_distance}-PRED-{int(time.time())}"
        tensorboard = TensorBoard(log_dir="logs/{}".format(name))

        history_fitting = self.model.fit(
            self.train_x, self.train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.test_x, self.test_y),
            callbacks=[tensorboard])
        return history_fitting

    def evaluate_visualize_model(self, history_records):
        score = self.model.evaluate(self.test_x, self.test_y, verbose=0)

        print('Validation mean_absolute_percentage_error:', score[1])
        print('Validation mse:', score[0])

        # Loss per epoch
        plt.plot(history_records.history['loss'])
        plt.plot(history_records.history['val_loss'])
        plt.title('Loss per epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

# Instantiate the class
pipeline = BitcoinPredictorPipeline(40, 2)

# Download the data
pipeline.download_data()
pipeline.preprocess_basic()
print("Training records: %s" % len(pipeline.train_x))
print("Validation records: %s" % len(pipeline.test_x))
pipeline.build_model()
history = pipeline.train_model()
pipeline.evaluate_visualize_model(history)
