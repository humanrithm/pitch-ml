import numpy as np
import pandas as pd
from typing import Union

# three-point central difference methods for derivative approximation 
    # Y: 2D array (m, n) or dataframe of shape (m, n) where m is the number of time series, n is the length
    # h: time step between points
def diff_three_point(
    Y: Union[np.ndarray | pd.DataFrame], 
    h: float = (1/480)
) -> np.ndarray:
    # option 1: numpy array
    if isinstance(Y, np.ndarray):
        # store the derivatives
        derivatives = np.zeros_like(Y)

        # handle boundary conditions using forward/backward difference
        derivatives[0, :] = (Y[1, :] - Y[0, :]) / h         # fwd difference at the start
        derivatives[-1, :] = (Y[-1, :] - Y[-2, :]) / h      # bwd difference at the end
        
        # compute the central difference for the interior points (ignoring the boundaries)
        derivatives[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / (2 * h)
    
    # option 2: dataframe
    elif isinstance(Y, pd.DataFrame):
        # store the derivatives
        derivatives = pd.DataFrame(np.zeros_like(Y), columns=Y.columns)

        # handle boundary conditions using forward/backward difference
        derivatives.iloc[0, :] = (Y.iloc[1, :] - Y.iloc[0, :]) / h
        derivatives.iloc[-1, :] = (Y.iloc[-1, :] - Y.iloc[-2, :]) / h

        # compute the central difference for the interior points (ignoring the boundaries)
        derivatives.iloc[1:-1, :] = (Y.iloc[2:, :].values - Y.iloc[:-2, :].values) / (2 * h)

        # update column names
        derivatives.columns = ['diff_' + col for col in derivatives.columns]
    
    return derivatives