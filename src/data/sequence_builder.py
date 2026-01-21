import numpy as np

def create_sequences(X, y, lookback):
    
    X_seq,y_seq = [],[]
    
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
        
    return np.array(X_seq), np.array(y_seq)
