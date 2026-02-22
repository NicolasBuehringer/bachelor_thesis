import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List
from tensorflow.keras.callbacks import EarlyStopping

from src.data_loader import load_raw_data_dic
from src.preprocessing import preprocess_dataframe, get_subsequence_images
from src.model import initialize_model, compile_model

def init_and_train_model(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray) -> Tuple[List, List]:
    '''
    Initialize, compile, and train a convolutional neural network model on the given training data,
    and evaluate it on the test data.
    '''
    # Initialize the model
    model = initialize_model()
    # Compile the model
    model = compile_model(model)
    # Set up early stopping callback
    es = EarlyStopping(patience=10, verbose=2)
    
    # Train the model with training data
    history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    callbacks=[es],
                    epochs=100,
                    batch_size=16, verbose=2)
    
    # Evaluate the model on test data
    evaluate = model.evaluate(X_test, y_test, verbose=0)

    # Generate predictions on the test data
    test_predict = model.predict(X_test, verbose=0)

    return [model, history], [evaluate, test_predict]

def prepare_and_train_cnn(df: pd.DataFrame) -> Tuple[List, List]:
    '''
    Preprocess the DataFrame, prepare training and test sets, normalize the data, 
    and train multiple CNN models.
    '''
    # Preprocess the DataFrame and split it into training and test sets
    train, test = preprocess_dataframe(df, split_size=0.85)

    # Get subsequence images for training and test sets as well as RV targets
    X_train, y_train = get_subsequence_images(train)
    X_test, y_test = get_subsequence_images(test)

    # Remove the last element from the training and test sets
    # since it doesn't have a Realized Volatility target
    if len(X_test) > 0:
        del X_test[-1]
    if len(X_train) > 0:
        del X_train[-1]

    # Convert the lists of subsequence images to numpy arrays
    # Normalize the image data to the range [0, 1]
    X_train_norm = np.array(X_train) / 255.
    X_test_norm = np.array(X_test) / 255.

    # Initialize and train CNN ensemble
    print("Training CNN 1")
    cnn_0 = init_and_train_model(X_train_norm, y_train, X_test_norm, y_test)
    print("Training CNN 2")
    cnn_1 = init_and_train_model(X_train_norm, y_train, X_test_norm, y_test)
    print("Training CNN 3")
    cnn_2 = init_and_train_model(X_train_norm, y_train, X_test_norm, y_test)
    print("Training CNN 4")
    cnn_3 = init_and_train_model(X_train_norm, y_train, X_test_norm, y_test)
    print("Training CNN 5")
    cnn_4 = init_and_train_model(X_train_norm, y_train, X_test_norm, y_test)

    # Collect results and models
    model_list = [cnn_0[0], cnn_1[0], cnn_2[0], cnn_3[0], cnn_4[0]]
    num_results_list = [cnn_0[1], cnn_1[1], cnn_2[1], cnn_3[1], cnn_4[1], X_test_norm, y_test]

    return model_list, num_results_list

def run_and_save_model(path: str, models_dir: str = 'all_models', results_dir: str = 'all_num_results') -> None:
    '''
    Load raw data from the specified path, train CNN models on each DataFrame, and save the models and results.
    '''
    counter = 1

    # Load raw data from the specified path into a dictionary of DataFrames
    dataframes_dic = load_raw_data_dic(path)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Iterate over each DataFrame in the dictionary
    for key, df in dataframes_dic.items():
        print(f"Processing {counter}: {key}")
        
        # Train CNN models and get the results
        model_dic, result_dic = prepare_and_train_cnn(df)

        # Save the trained models to a file
        with open(os.path.join(models_dir, f'{key}.pkl'), 'wb') as f:
            pickle.dump(model_dic, f)

        # Save the results to a file
        with open(os.path.join(results_dir, f'{key}.pkl'), 'wb') as f:
            pickle.dump(result_dic, f)
    
        counter += 1
