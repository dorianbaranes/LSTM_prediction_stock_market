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
import multiprocessing

def analysis_with_test_data(model_prop, x_test, y_test, window_size, prediction_len):
    model = Model()
    model.load_model(model_prop['path_model'])
    predictions = model.predict_sequences_multiple(x_test, window_size, prediction_len)

    # Calculate RMSE for test predictions
    all_predictions = [item for window in predictions for item in window]
    test_rmse = np.sqrt(
        mean_squared_error(y_test, all_predictions[: len(y_test)])
    )

    # Calculate error margin on trend direction
    score_trend, nb_predictions = calculate_trend(y_test, predictions)

    return ({
        **{
            "RMSE": test_rmse,
            "score_trend": score_trend,
            "nb_predictions": nb_predictions,
        },
        **model_prop,
    })



def plot_results_multiple(predicted_data,most_recent_predicted_data, true_data, prices, prediction_len, quality_information):

    fig = plt.figure(facecolor="white")
    ax1 = fig.add_subplot(111)
    ax1.plot(true_data, label="Nasdaq-100 Index", alpha=0.5)
    # Pad the list of predictions to shift it in the graph to it's correct start

    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        ax1.plot(padding + data, label="Prediction", color='fuchsia', alpha=0.8)

        for y in range(1, len(data)):
            if data[y] < data[y - 1]:
                plt.axvspan(len(padding)+y - 1, len(padding)+y, facecolor='red', alpha=0.1)
            elif data[y] > data[y - 1]:
                plt.axvspan(len(padding)+y - 1, len(padding)+y, facecolor='green', alpha=0.1)

    padding = [None for p in range(len(true_data))]
    offset_x=len(padding)

    alpha_values = np.linspace(1, 0, int(len(most_recent_predicted_data)/2))
    alpha_idx=0
    for i in range(len(most_recent_predicted_data)-1):
        if i<int(len(most_recent_predicted_data)/2):
            ax1.plot([offset_x + i, offset_x + i + 1], most_recent_predicted_data[i:i + 2], color='fuchsia',alpha=1)
        else:
            ax1.plot([offset_x+i,offset_x+i+1], most_recent_predicted_data[i:i+2], color='fuchsia', alpha=alpha_values[alpha_idx])
            alpha_idx += 1

    #ax1.plot(padding + most_recent_predicted_data, color='black', alpha=alpha_values)


    ax2 = ax1.twinx()
    ax2.plot(prices, color="lightblue", marker="o", linewidth=0.5)


    # Move ax1 to the foreground
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    plt.title( "Trend Prediction of the Nasdaq-100 Index")
    plt.text(0.05, 0.05, quality_information, transform=plt.gca().transAxes, ha='left', va='top')
    plt.legend()


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

    min_batch_size = configs["training"]["batch_size"]["min"]
    max_batch_size = configs["training"]["batch_size"]["max"]
    step_batch_size = configs["training"]["batch_size"]["step"]

    models_properties = list()

    for batch_size in range(min_batch_size, max_batch_size + 1, step_batch_size):
        model.model.reset_states()
        data.gen_data_generators(archi_model["sequence_length"], configs["data"]["validation_train_split"], batch_size)
        steps_per_epoch = math.ceil(
            (data.len_train - archi_model["sequence_length"]) / batch_size
        )


        for epoch in range(min_epoch, max_epoch + 1):
            # out-of memory generative training
            file_name_prefix = f"{dt.datetime.now().strftime('%d%m%Y_%H%M%S')}_epc_{epoch}_batch_{batch_size}"

            train_loss,val_loss=model.train_generator(
                data_train_gen=data.data_train_generator,
                data_val_gen=data.data_val_generator,
                epochs=(min_epoch if epoch == min_epoch else 1),
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                save_dir=os.path.join(configs["model"]["save_dir"],'models'),
                file_name_prefix=file_name_prefix,
            )


            path_model = os.path.join(configs["model"]["save_dir"],'models',file_name_prefix+"_model.h5")
            model_prop={
                    **{
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "dif_val_train_loss": ((val_loss - train_loss) / train_loss) * 100,
                        "model_name": file_name_prefix + "_model.h5",
                        "path_model" : path_model
                    },
                    **archi_model,
                }

            models_properties.append(model_prop)

    return models_properties


def build_models():
    configs = json.load(open("config.json", "r"))

    if not os.path.exists(os.path.join(configs["model"]["save_dir"],'props')):
        os.makedirs(os.path.join(configs["model"]["save_dir"],'props'))

    if not os.path.exists(os.path.join(configs["model"]["save_dir"],'models')):
        os.makedirs(os.path.join(configs["model"]["save_dir"],'models'))

    # Save JSON string to a file
    model_properties_path = os.path.join(
        configs["model"]["save_dir"],
        'props',
        f"{configs['data']['filename']}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}_model_properties.json",
    )


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

    min_seq_len = configs["data"]["sequence_length"]["min"]
    max_seq_len = configs["data"]["sequence_length"]["max"]
    step_seq_len = configs["data"]["sequence_length"]["step"]

    props = {
        "configs": configs,
        "models": list()
    }

    with Timer() as timer:
        for sequence_length in range(min_seq_len, max_seq_len+1, step_seq_len):
            x_test, y_test = data.get_test_data(
                seq_len=sequence_length
            )
            for lstm_layer in range(min_lstm_layer, max_lstm_layer + 1, step_lstm_layer):
                for neurons in range(
                    min_neurons_layer, max_neurons_layer + 1, step_neurons_layer
                ):

                    archi_model = {
                        "neurons_per_lstm_layer": neurons,
                        "nb_lstm_layer": lstm_layer,
                        "sequence_length": sequence_length
                    }

                    # Create a Queue for communication between processes
                    output_queue = multiprocessing.Queue()
                    training_process = multiprocessing.Process(target=build_train_model, args=(archi_model, configs, data, output_queue))
                    training_process.start()
                    training_process.join()
                    # Get the result from the output queue
                    models_props = output_queue.get()
                    print(models_props)

                    '''
                    # Number of parallel processes
                    num_processes = multiprocessing.cpu_count()
                    pool = multiprocessing.Pool(processes=num_processes)
                    model_prop_results = list()
                    for model_prop in models_props:
                        result = pool.apply_async(analysis_with_test_data, (
                            model_prop,
                            x_test,
                            y_test,
                            sequence_length,
                            configs["data"]["prediction_length"]
                        ))
                        model_prop_results.append(result)



                    # Wait for all processes to finish and collect the results
                    print(f"Wait for analysis with test data. Number of analysis:{len(model_prop_results)}")
                    models_props = [result.get() for result in model_prop_results]


                    # Close the pool of workers
                    pool.close()
                    pool.join()
                    '''

                    props['models'] += models_props
                    props['models'] = sorted(props['models'], key=lambda x: (x['train_loss'], x['val_loss']))



                    with open(model_properties_path, "w") as file:
                        json.dump(props, file, indent=4)

    return model_properties_path


def build_train_model(archi_model, configs, data, output_queue):
    model = Model()
    neurons=archi_model["neurons_per_lstm_layer"]
    lstm_layer=archi_model["nb_lstm_layer"]
    sequence_length = archi_model["sequence_length"]
    model.build_model(configs, neurons, lstm_layer, sequence_length)
    props=train_model(configs, model, data, archi_model)
    del model.model
    output_queue.put(props)


def check_model(model_filename):
    props_dir = os.path.join('saved_models', 'props')

    prop_files = os.listdir(props_dir)

    # Open and read each file
    config=None
    for file_name in prop_files:
        file_path = os.path.join(props_dir, file_name)

        with open(file_path, 'r') as file:
            props = json.load(file)
            for current_model in props["models"]:
                if model_filename == current_model["model_name"]:
                    configs=props["configs"]
                    model_config=current_model
                    break
        if config:
            break

    data = DataLoader(
        os.path.join("data", configs["data"]["filename"]),
        configs["data"]["train_test_split"],
        configs["data"]["columns"],
        configs["data"]["from_date"],
    )

    path_model = os.path.join(configs["model"]["save_dir"],'models', model_filename)
    model = Model()
    model.load_model(path_model)

    x_test, y_test = data.get_test_data(seq_len=model_config["sequence_length"])

    predictions = model.predict_sequences_multiple(
        x_test, model_config["sequence_length"], configs["data"]["prediction_length"]
    )

    x_test_last_seq=[x_test[-1]]
    most_recent_prediction= model.predict_sequences_multiple(x_test_last_seq, model_config["sequence_length"], model_config["sequence_length"])[0]

    # Calculate RMSE for test predictions
    all_predictions = [item for window in predictions for item in window]
    test_rmse = np.sqrt(mean_squared_error(y_test, all_predictions[: len(y_test)]))
    print("Test RMSE:", test_rmse)

    # Calculate error margin on trend direction
    score_trend, nb_predictions = calculate_trend(y_test, predictions)
    print(f"score_trend:{score_trend} nb_predictions:{nb_predictions}")

    quality_info=f"RMSE:{round(test_rmse,3)} Prediction True: {round(score_trend,3)}%"
    prices = data.data_test[model_config["sequence_length"] - 1 :, 0]
    plot_results_multiple(
        predictions,most_recent_prediction, y_test, prices, configs["data"]["prediction_length"],quality_info
    )





if __name__ == "__main__":


    while True:
        print("\nMenu:")
        print("1. Run a model")
        print("2. Build models")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            model_name = input("Enter model name (eg. 30052023_203358_epc_6_batch_120_model.h5): ")
            check_model(model_name)
        elif choice == '2':
            path_prop_file_name = build_models()


            with open(path_prop_file_name) as f:
                data = json.load(f)

            # check model with the best RMSE
            check_model(data['models'][0]['model_name'])
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter a valid option.")






