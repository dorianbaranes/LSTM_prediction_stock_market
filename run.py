__author__ = "Dorian Baranes"
__copyright__ = "Dorian Baranes 2023"
__version__ = "1.0.0"
__license__ = "MIT"

import os
import json
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
from core.utils import Timer
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime as dt


def plot_results_multiple(predicted_data, true_data, prices, prediction_len, quality_information):

    fig = plt.figure(facecolor="white")
    ax1 = fig.add_subplot(111)
    ax1.plot(true_data, label="True Data")
    # Pad the list of predictions to shift it in the graph to it's correct start

    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        color = "red" if data[0] >= data[-1] else "green"
        ax1.plot(padding + data, label="Prediction", color=color)

    ax2 = ax1.twinx()
    ax2.plot(prices, label="Prices", color="lightblue", marker="o", linewidth=0.5)

    # Add vertical lines for each x-value
    for val in range(int(len(prices) / 10) + 1):
        ax1.axvline(val * 10, color="lightgray", linestyle="--", linewidth=0.5)
        ax2.axvline(val * 10, color="lightgray", linestyle="--", linewidth=0.5)

    # Move ax1 to the foreground
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    plt.title( "Trend Prediction of the Nasdaq-100 Index")
    plt.text(0.05, 0.05, quality_information, transform=plt.gca().transAxes, ha='left', va='top')


    plt.show()


def calculate_trend(test_data, forecast):
    # calculate trend_prediction_level:
    len_window = len(forecast[0])
    score = 0
    nb_predictions = 0
    for i, window in enumerate(forecast):
        if len(test_data) <= (i * len_window + len_window):
            break
        nb_predictions += 1
        forecast_trend_up = True if window[0] < window[-1] else False
        test_trend_up = (
            True
            if test_data[i * len_window] < test_data[i * len_window + len_window]
            else False
        )

        if forecast_trend_up == test_trend_up:
            score += 1

    score_trend = (float(score / nb_predictions)) * 100.0

    return (score_trend, nb_predictions)


def train_model(configs, model, data, archi_model):
    min_epoch = configs["training"]["epochs"]["min"]
    max_epoch = configs["training"]["epochs"]["max"]
    step_epoch = configs["training"]["epochs"]["step"]

    min_batch_size = configs["training"]["batch_size"]["min"]
    max_batch_size = configs["training"]["batch_size"]["max"]
    step_batch_size = configs["training"]["batch_size"]["step"]

    models_properties = dict()

    for batch_size in range(min_batch_size, max_batch_size + 1, step_batch_size):
        model.model.reset_states()
        steps_per_epoch = math.ceil(
            (data.len_train - configs["data"]["sequence_length"]) / batch_size
        )

        for epoch in range(min_epoch, max_epoch + 1, step_epoch):
            # out-of memory generative training
            file_name_prefix = f"{dt.datetime.now().strftime('%d%m%Y_%H%M%S')}_epc_{epoch}_batch_{batch_size}"

            model.train_generator(
                data_gen=data.generate_train_batch(
                    seq_len=configs["data"]["sequence_length"], batch_size=batch_size
                ),
                epochs=(min_epoch if epoch == min_epoch else 1),
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                save_dir=configs["model"]["save_dir"],
                file_name_prefix=file_name_prefix,
            )

            x_test, y_test = data.get_test_data(
                seq_len=configs["data"]["sequence_length"]
            )

            predictions = model.predict_sequences_multiple(
                x_test,
                configs["data"]["sequence_length"],
                configs["data"]["prediction_length"],
            )

            # Calculate RMSE for test predictions
            all_predictions = [item for window in predictions for item in window]
            test_rmse = np.sqrt(
                mean_squared_error(y_test, all_predictions[: len(y_test)])
            )
            print("Test RMSE:", test_rmse)

            # Calculate error margin on trend direction
            score_trend, nb_predictions = calculate_trend(y_test, predictions)
            print(f"score_trend:{score_trend} nb_predictions:{nb_predictions}")

            models_properties[file_name_prefix] = {
                **{
                    "RMSE": test_rmse,
                    "score_trend": score_trend,
                    "nb_predictions": nb_predictions,
                    "model": file_name_prefix + "_model.h5",
                },
                **archi_model,
            }

    return models_properties


def main():
    configs = json.load(open("config.json", "r"))
    if not os.path.exists(configs["model"]["save_dir"]):
        os.makedirs(configs["model"]["save_dir"])

    # Save JSON string to a file
    model_properties_path = os.path.join(
        configs["model"]["save_dir"],
        f"{dt.datetime.now().strftime('%d%m%Y_%H%M%S')}_model_properties.json",
    )

    all_models_properties = dict()

    configs = json.load(open("config.json", "r"))
    if not os.path.exists(configs["model"]["save_dir"]):
        os.makedirs(configs["model"]["save_dir"])

    data = DataLoader(
        os.path.join("data", configs["data"]["filename"]),
        configs["data"]["train_test_split"],
        configs["data"]["columns"],
        configs["data"]["from_date"],
    )

    min_neurons_layer = configs["model"]["neurons_per_lstm_layer"]["min"]
    max_neurons_layer = configs["model"]["neurons_per_lstm_layer"]["max"]
    step_neurons_layer = configs["model"]["neurons_per_lstm_layer"]["step"]

    min_lstm_layer = configs["model"]["lstm_layers"]["min"]
    max_lstm_layer = configs["model"]["lstm_layers"]["max"]
    step_lstm_layer = configs["model"]["lstm_layers"]["step"]

    with Timer() as timer:
        for lstm_layer in range(min_lstm_layer, max_lstm_layer + 1, step_lstm_layer):
            for neurons in range(
                min_neurons_layer, max_neurons_layer + 1, step_neurons_layer
            ):
                archi_model = {
                    "neurons_per_lstm_layer": neurons,
                    "nb_lstm_layer": lstm_layer,
                }
                model = Model()
                model.build_model(configs, neurons, lstm_layer)

                models_props = train_model(configs, model, data, archi_model)
                all_models_properties = {**models_props, **all_models_properties}

                with open(model_properties_path, "w") as file:
                    json.dump(all_models_properties, file, indent=4)


def check_model(filename):
    configs = json.load(open("config.json", "r"))
    if not os.path.exists(configs["model"]["save_dir"]):
        os.makedirs(configs["model"]["save_dir"])

    data = DataLoader(
        os.path.join("data", configs["data"]["filename"]),
        configs["data"]["train_test_split"],
        configs["data"]["columns"],
        configs["data"]["from_date"],
    )

    path_model = os.path.join(configs["model"]["save_dir"], filename)
    model = Model()
    model.load_model(path_model)

    x_test, y_test = data.get_test_data(seq_len=configs["data"]["sequence_length"])

    predictions = model.predict_sequences_multiple(
        x_test, configs["data"]["sequence_length"], configs["data"]["prediction_length"]
    )

    # Calculate RMSE for test predictions
    all_predictions = [item for window in predictions for item in window]
    test_rmse = np.sqrt(mean_squared_error(y_test, all_predictions[: len(y_test)]))
    print("Test RMSE:", test_rmse)

    # Calculate error margin on trend direction
    score_trend, nb_predictions = calculate_trend(y_test, predictions)
    print(f"score_trend:{score_trend} nb_predictions:{nb_predictions}")

    quality_info=f"RMSE:{round(test_rmse*100,3)}% Prediction True: {round(score_trend,3)}%"
    prices = data.data_test[configs["data"]["sequence_length"] - 1 :, 0]
    plot_results_multiple(
        predictions, y_test, prices, configs["data"]["prediction_length"],quality_info
    )


def sort_model_RMSE(filename):
    configs = json.load(open("config.json", "r"))
    if not os.path.exists(configs["model"]["save_dir"]):
        os.makedirs(configs["model"]["save_dir"])

    path_models = os.path.join(configs["model"]["save_dir"], filename)

    # Read the JSON file
    with open(path_models) as f:
        data = json.load(f)

    # Convert the dictionary into a list
    data_list = list(data.values())
    data_list = [{**{"prefix": key}, **value} for key, value in data.items()]

    # Sort the list by the "rmse" key
    sorted_list = sorted(data_list, key=lambda x: x["RMSE"])

    # Write the sorted list to a JSON file
    path_sort_models = os.path.join(
        configs["model"]["save_dir"], "sorted_RMSE_model.json"
    )
    with open(path_sort_models, "w") as f:
        json.dump(sorted_list, f, indent=4)


def sort_model_trend(filename):
    configs = json.load(open("config.json", "r"))
    if not os.path.exists(configs["model"]["save_dir"]):
        os.makedirs(configs["model"]["save_dir"])

    path_models = os.path.join(configs["model"]["save_dir"], filename)

    # Read the JSON file
    with open(path_models) as f:
        data = json.load(f)

    # Convert the dictionary into a list
    data_list = list(data.values())
    data_list = [{**{"prefix": key}, **value} for key, value in data.items()]

    # Sort the list by the "rmse" key
    sorted_list = sorted(data_list, key=lambda x: x["score_trend"])

    # Write the sorted list to a JSON file
    path_sort_models = os.path.join(
        configs["model"]["save_dir"], "sorted_score_trend_model.json"
    )
    with open(path_sort_models, "w") as f:
        json.dump(sorted_list, f, indent=4)


if __name__ == "__main__":
    #main()
    check_model("28052023_121818_epc_6_batch_90_model.h5")
    #sort_model_RMSE("28052023_120758_model_properties.json")
    #sort_model_trend("28052023_120758_model_properties.json")
