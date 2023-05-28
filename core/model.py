import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# Needed for reproducible results
np.random.seed(1)


class Model:
    def __init__(self):
        self.model = Sequential()
        self.neurons_per_lstm_layer = None
        self.lstm_layers = None

    def load_model(self, filepath):
        print("[Model] Loading model from file %s" % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs, neurons_lstm, layer_lstm):
        self.neurons_per_lstm_layer = neurons_lstm
        self.lstm_layers = layer_lstm

        for layer in configs["model"]["layers"]:
            neurons = layer["neurons"] if "neurons" in layer else None
            dropout_rate = layer["rate"] if "rate" in layer else None
            activation = layer["activation"] if "activation" in layer else None
            return_seq = layer["return_seq"] if "return_seq" in layer else None
            input_timesteps = (
                layer["input_timesteps"] if "input_timesteps" in layer else None
            )
            input_dim = layer["input_dim"] if "input_dim" in layer else None

            if layer["type"] == "dense":
                self.model.add(Dense(neurons, activation=activation))
            if layer["type"] == "lstm":
                self.model.add(
                    LSTM(
                        neurons_lstm,
                        input_shape=(input_timesteps, input_dim),
                        return_sequences=return_seq,
                    )
                )
            if layer["type"] == "dropout":
                self.model.add(Dropout(dropout_rate))
            if layer["type"] == "hidden_lstm_layers":
                for i in range(layer_lstm - 1):
                    self.model.add(LSTM(neurons_lstm, return_sequences=True))
                    self.model.add(Dropout(0.2))
                self.model.add(LSTM(neurons_lstm, return_sequences=False))
                self.model.add(Dropout(0.2))

        self.model.compile(
            loss=configs["model"]["loss"], optimizer=configs["model"]["optimizer"]
        )

        print("[Model] Model Compiled")

    def train_generator(
        self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, file_name_prefix
    ):
        print("[Model] Training Started")
        print(
            "[Model] %s epochs, %s batch size, %s batches per epoch %s lstm layers %s neurons per layer"
            % (
                epochs,
                batch_size,
                steps_per_epoch,
                self.lstm_layers,
                self.neurons_per_lstm_layer,
            )
        )
        save_fname_weight = os.path.join(save_dir, file_name_prefix + "_weight.h5")
        save_fname_model = os.path.join(save_dir, file_name_prefix + "_model.h5")

        callbacks = [
            # ModelCheckpoint(filepath=save_fname_weight, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1,
        )
        self.model.save(save_fname_model)
        print("[Model] Training Completed. Model saved as %s" % save_fname_weight)

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print("[Model] Predicting Point-by-Point...")
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print("[Model] Predicting Sequences Multiple...")
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len) + 1):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(
                    curr_frame, [window_size - 2], predicted[-1], axis=0
                )
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print("[Model] Predicting Sequences Full...")
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
