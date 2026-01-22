from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape):
    
    '''
    Build a simple LSTM model for time-series regression
    
    Parameters:
        input_shape (tuple): (time_steps, features)
        
    Returns:
        model(keres.Model)
        
    '''
    
    model = Sequential()
    
    model.add(
        LSTM(
            units = 50,
            activation = "tanh",
            input_shape = input_shape
        )
    )
    
    model.add(
        Dense(1)
    )
    
    model.compile(
        optimizer = Adam(learning_rate=0.001),
        loss="mse"
    )
    
    return model