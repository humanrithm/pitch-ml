import numpy as np
import pandas as pd
from typing import Union
from scipy.signal import butter, filtfilt


__version__ = '0.2.0'

def butter_lowpass_filter(
        data: pd.DataFrame, 
        columns: Union[list, str], 
        cutoff: float = 18, 
        fs: float = 480.0, 
        order: int = 4
) -> pd.DataFrame:
    """
    Apply a zero-phase Butterworth low-pass filter to biomechanics marker data.

    **Args**:
    `data` (pd.DataFrame): Input DataFrame with marker position columns.
    `columns` (str or list of str): Column(s) to filter.
    `cutoff` (float): Cutoff frequency in Hz. Default is 13.4 Hz.
    `fs` (float): Sampling frequency in Hz. Default is 480 Hz.
    `order` (int): Order of the Butterworth filter. Default is 4.

    Returns:
    `pd.DataFrame`: Filtered DataFrame with the same index and filtered columns.
    """
    
    # 
    if isinstance(columns, str):
        columns = [columns]

    # filtering params
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False)

    # setup filtered df
    filtered_df = data.copy()

    # iterate through columns
    for col in columns:
        x = data[col].values
        x_filtered = filtfilt(b, a, x, method="pad")
        filtered_df[f'{col}_filtered'] = x_filtered

    return filtered_df