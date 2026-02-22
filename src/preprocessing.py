import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None

def preprocess_dataframe(data: pd.DataFrame, split_size: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Preprocess the input DataFrame by calculating log returns and realized volatility, 
    and split it into training and testing sets.
    '''
    df = data.copy()
    df["Date Time"] = pd.to_datetime(df["Date Time"])

    # calculate log returns
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # remove every first minute of each day
    df = df[~(df['Date Time'].dt.time == pd.Timestamp('09:35').time())].reset_index()
    
    # Create a flag to group by days (380 rows)
    df["day"] = (df.index // 380)

    # Calculate the RV for each grouped 380 row day
    group_sums = df.groupby("day")["Log_Returns"].transform(lambda x: np.sqrt(np.sum(x**2)))
    
    # Add the group sums as a new column
    df["realized_vol"] = group_sums
    
    df.set_index("Date Time", inplace=True)

    # calculate train test split; make sure no day is cut in half between the two data splits
    split_index = int(np.floor(df.shape[0] * split_size / 380) * 380)

    # Split the DataFrame into training and testing sets
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    return df_train, df_test

def create_subsequences(dataframe: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[float]]:
    '''
    Create subsequences for training and a list of RV targets
    '''
    # create lists to fill with subsequences and their Realized Volatility targets
    sequence_list = []
    sequence_target = []

    # Loop through the DataFrame to create subsequences
    for i in range(int(len(dataframe)/380-20)):
        
        # Extract a subsequence of 21 days (each day with 380 rows)
        tmp_subsequence = dataframe.iloc[i * 380: (21 + i) * 380]
        sequence_list.append(tmp_subsequence)

        # Try to get the target value for the current subsequence
        try:
            subsequence_target = dataframe.iloc[(21 + i) * 380]["realized_vol"]
            sequence_target.append(subsequence_target)
        except:
            # Print a message if an IndexError occurs (likely in the last iteration)
            # since the last 21 day window doesn't have a RV target
            print("last iteration")

    return sequence_list, sequence_target

def one_month_to_image(one_month: np.ndarray, add_blue: bool = False) -> np.ndarray:
    '''
    Convert a month's worth of log returns data into image representation of size 21x380x2
    If images are needed for plotting, add blue channel and return 21x380x3 images
    '''
    size = len(one_month)

    # Initialize arrays for the RGB channels; should be (7980,)
    red = np.zeros(size)
    green = np.zeros(size)
    blue = np.zeros(size)

    # adding negative returns absolute values to red channel
    # and positive returns to green channel
    # fill other channel with zero
    
    # leave blue channel at zeros
    for i in range(len(one_month)):
        log_return = one_month[i]

        if log_return < 0:
            red[i] = abs(log_return)
        elif log_return > 0:
            green[i] = log_return
        elif log_return == 0:
            continue

    # turning shape (7980,) into (7980,1) for scaling
    red = np.reshape(np.array(red), (-1,1))
    green = np.reshape(np.array(green), (-1,1))
    
    # scaling returns to [0, 255] interval
    mm_scaler_red = MinMaxScaler(feature_range=(0,255))
    red_scaled = mm_scaler_red.fit_transform(red)

    mm_scaler_green = MinMaxScaler(feature_range=(0,255))
    green_scaled = mm_scaler_green.fit_transform(green)

    # Flatten the scaled arrays back to (size,) to stack
    red_scaled_flat = red_scaled.flatten()
    green_scaled_flat = green_scaled.flatten()

    # flag if blue color channel is needed for plotting
    if add_blue:
        # Stack the red, green, and blue channels into a single array
        flat_image = np.column_stack((red_scaled_flat,
                                      green_scaled_flat,
                                      blue))
         # Reshape the array to (21, 380, 3) for image representation
        square_image = flat_image.reshape((21, 380, 3))

    # don't add blue channel if images are used in CNN
    else:
        # Stack the red and green channels into a single array
        flat_image = np.column_stack((red_scaled_flat,
                                      green_scaled_flat))
    
        # Reshape the array to (21, 380, 2) for CNN training
        square_image = flat_image.reshape((21, 380, 2))

    return square_image

def create_images(sequence_list: List[pd.DataFrame]) -> List[np.ndarray]:
    '''
    Convert a list of DataFrame subsequences each containin 21 days
    into a list of their image representations
    '''
    image_list = []

    # Loop through each rolling month DataFrame in the sequence list
    for rolling_month in sequence_list:
        
        # Extract the 'Log_Returns' column as a numpy array
        X_array = np.array(rolling_month["Log_Returns"])

        # Convert each month into its image representation
        image = one_month_to_image(X_array)

        image_list.append(image)

    return image_list

def get_subsequence_images(dataframe: pd.DataFrame) -> Tuple[List[np.ndarray], np.ndarray]:
    '''
    Helper function: Generate image representations for subsequences of log returns from the input DataFrame.
    '''
    # Create subsequences and their RV targets from the input DataFrame
    month_sequences, subsequence_targets = create_subsequences(dataframe)

     # Generate images for the created subsequences
    image_list = create_images(month_sequences)

    # Convert the list of target values to a numpy array and return the images and their targets
    return image_list, np.array(subsequence_targets)
