from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam

def initialize_model() -> Sequential:
    '''
    Initialize and return a Sequential convolutional neural network model.
    '''
    model = Sequential()
    model.add(Input(shape=(21, 380, 2)))
    
    model.add(Conv2D(16, (3,5), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(1, activation="linear"))

    return model

def compile_model(model: Sequential, learning_rate: float = 0.0001) -> Sequential:
    '''
    Compile the given Sequential model with mean squared error loss and Adam optimizer.
    '''
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss='mse', optimizer=optimizer)
    return model
