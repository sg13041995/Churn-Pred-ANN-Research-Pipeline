from tensorflow.keras.models import model_from_json
from contextlib import redirect_stdout
import json
import shutil
import os
import warnings
warnings.filterwarnings("ignore")


def trainable_parameters_count(layers):

    params = {
        "input params":  layers["input"],
        "hidden 1 params": (layers["input"] * layers["hidden 1"]) + layers["hidden 1"],
        "output params": 0,
    }

    all_layers = set(layers.keys())
    remaining_hidden_layers = set()

    for layer_keys in layers.keys():
        all_layers.add(layer_keys)

    remaining_hidden_layers = all_layers - set(("input", "output", "hidden 1"))

    # print(all_layers, hidden_layers)

    if (len(remaining_hidden_layers) > 0):
        for i, _ in enumerate(remaining_hidden_layers, 2):
            params[f"hidden {i} params"] = (
                layers[f"hidden {i-1}"] * layers[f"hidden {i}"]) + layers[f"hidden {i}"]

    params["output params"] = (
        layers[f"hidden {len(remaining_hidden_layers) + 1}"] * layers["output"]) + layers["output"]

    print(params)
    print(
        f"Total trainable params = {(sum(params.values())) - (params['input params'])}")

def import_export_model(model,
                        model_path="model_1.json",
                        params_path="model_1_params.h5",
                        is_export=True,
                        is_import=False,
                        architecture=True,
                        params=True):

    if is_export:
        if architecture:
            # keras model to json
            model_json = model.to_json()
            # export json into a file
            with open(model_path, "w") as json_file:
                json_file.write(model_json)

        if params:
            # serialize weights to HDF5
            model.save_weights(params_path)

    if is_import:
        if architecture:
            # load json and create model
            with open(model_path, 'r') as json_file:
                loaded_model_json = json_file.read()

            model = model_from_json(loaded_model_json)

        if params:
            # load weights into new model
            model.load_weights("model_params.h5")

        return model


def folder_exists(directory, folder_name):
    folder_path = os.path.join(directory, folder_name)
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


def zip_n_download(directory, folder_name, zip_file_name):
    folder_path = os.path.join(directory, folder_name)

    # Zip the folder
    shutil.make_archive(zip_file_name, 'zip', folder_path)

    if os.getcwd() == "/content":
        # Download the zip file
        files.download(f"{zip_file_name}.zip")


def export_arch_params(CURRENT_DIR, new_dir, model_number, model, is_export=True, is_import=False):

    if (os.getcwd() == CURRENT_DIR):
        os.chdir(new_dir)

        model_path = f"model_{model_number}_architectute.json"
        params_path = f"model_{model_number}_params.h5"

        import_export_model(model,
                            model_path=model_path,
                            params_path=params_path,
                            is_export=is_export,
                            is_import=is_import)

        os.chdir("..")

    else:
        print("Current directory not matching...")
        print("Model won't be exported")
        print(f"Present working directory: {os.getcwd()}")

    if (os.getcwd() == CURRENT_DIR):
        print("You are back to Home directory...")
    else:
        print("WARNING!!!")
        print("You are not in your home directory")
        print(f"Present working directory: {os.getcwd()}")


def export_history_json(CURRENT_DIR, new_dir, model_number, history_data):
    history_file_path = f"model_{model_number}_history.json"
    data = history_data

    if (os.getcwd() == CURRENT_DIR):
        os.chdir(new_dir)

        # Write the Model History to a JSON file
        with open(history_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        os.chdir("..")
    else:
        print("Current directory not matching...")
        print("History won't be exported")
        print(f"Present working directory: {os.getcwd()}")

    if (os.getcwd() == CURRENT_DIR):
        print("You are back to Home directory...")
    else:
        print("WARNING!!!")
        print("You are not in your home directory")
        print(f"Present working directory: {os.getcwd()}")


def export_model_config(CURRENT_DIR, new_dir, model_number, model_config_profile_data):

    configuration_file_path = f"model_{model_number}_config.json"
    data = model_config_profile_data

    if (os.getcwd() == CURRENT_DIR):
        os.chdir(new_dir)

        # Write the Model History to a JSON file
        with open(configuration_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        os.chdir("..")
    else:
        print("Current directory not matching...")
        print("Configuration won't be exported")
        print(f"Present working directory: {os.getcwd()}")

    if (os.getcwd() == CURRENT_DIR):
        print("You are back to Home directory...")
    else:
        print("WARNING!!!")
        print("You are not in your home directory")
        print(f"Present working directory: {os.getcwd()}")


def export_model_summary(CURRENT_DIR, new_dir, model_number, model):
    # Define file path for saving the summary
    summary_file = f"model_{model_number}_summary.txt"

    if (os.getcwd() == CURRENT_DIR):
        os.chdir(new_dir)

        # Open a file for writing the summary
        with open(summary_file, "w") as f:
            # Redirect stdout to the file
            with redirect_stdout(f):
                # Print the model summary
                model.summary()
        os.chdir("..")

    else:
        print("Current directory not matching...")
        print("Model summary won't be exported")
        print(f"Present working directory: {os.getcwd()}")

    if (os.getcwd() == CURRENT_DIR):
        print("You are back to Home directory...")
    else:
        print("WARNING!!!")
        print("You are not in your home directory")
        print(f"Present working directory: {os.getcwd()}")


def export_model_update_details(CURRENT_DIR, new_dir, model_number, model_update_details):
    # Define file path for saving the summary
    model_update_details_file = f"model_{model_number}_update_details.txt"

    if (os.getcwd() == CURRENT_DIR):
        os.chdir(new_dir)

        # Open a file for writing the summary
        with open(model_update_details_file, "w") as f:
            f.write(model_update_details)
        os.chdir("..")

    else:
        print("Current directory not matching...")
        print("Model update details won't be exported")
        print(f"Present working directory: {os.getcwd()}")

    if (os.getcwd() == CURRENT_DIR):
        print("You are back to Home directory...")
    else:
        print("WARNING!!!")
        print("You are not in your home directory")
        print(f"Present working directory: {os.getcwd()}")
