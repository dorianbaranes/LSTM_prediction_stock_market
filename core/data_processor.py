import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
class DataLoader():
    def __init__(self, filename, split, cols, from_date):
        dataframe = pd.read_csv(filename,  index_col='Date')
        dataframe = dataframe.loc[from_date:]

        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

        # Train the Scaler with training data and smooth data
        scaler = MinMaxScaler(feature_range=(0, 1))
        smoothing_window_size = 1000

        # Iterate over the dataframe in chunks of 500 rows
        for i in range(0, len(self.data_train), smoothing_window_size):
            chunk = self.data_train[i:i + smoothing_window_size]  # Get the chunk of 500 rows
            self.data_train[i:i + smoothing_window_size] = scaler.fit_transform(chunk)

        # Iterate over the dataframe in chunks of 500 rows
        for i in range(0, len(self.data_test), smoothing_window_size):
            chunk = self.data_test[i:i + smoothing_window_size]  # Get the chunk of 500 rows
            self.data_test[i:i + smoothing_window_size] = scaler.fit_transform(chunk)

        #smooth the data smooth the data using the exponential moving average.
        #This helps you to get rid of the inherent raggedness of the data in stock prices and produce a
        #smoother curve.
        for col_idx in range(len(cols)):
            EMA = 0.0
            gamma = 0.1
            for row_idx in range(self.data_train.shape[0]):
              EMA = gamma*self.data_train[row_idx,col_idx] + (1-gamma)*EMA
              self.data_train[row_idx,col_idx] = EMA

    def get_test_data(self, seq_len):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y
