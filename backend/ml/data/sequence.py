# backend/ml/data/sequence.py

import numpy as np
import pandas as pd
from typing import Tuple


def make_sequences(
    df: pd.DataFrame,
    target_col: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a feature DataFrame into sequences for time-series modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (no NaNs).
    target_col : str
        Column name to predict.
    seq_len : int
        Number of past timesteps per sequence.

    Returns
    -------
    X : np.ndarray
        Shape (num_samples, seq_len, num_features)
    y : np.ndarray
        Shape (num_samples,)
    """

    values = df.to_numpy()
    target_idx = df.columns.get_loc(target_col)

    X, y = [], []

    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len : i]) #This takes the entire row at a specific time period
        y.append(values[i, target_idx])  #This is basically taking the target columns value at the ith row/ith time index.

    return np.array(X), np.array(y) #Now, X holds all these rows, and y holds the target col values. We can then use this along with
#the sequence length to go ahead with supervised learning. 


