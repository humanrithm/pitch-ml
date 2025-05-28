import numpy as np
import pandas as pd
from typing import Union
from scipy.signal import butter, filtfilt
from .diff_three_point import diff_three_point


__version__ = '0.1.7'

def butter_lowpass(
        cutoff: float, 
        fs: float, 
        order: int
    ):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    return b, a    

def butter_lowpass_filter(
        data: Union[pd.DataFrame, np.ndarray], 
        cutoff: float = 13.4, 
        fs: int = 480, 
        order: int = 4,
    ):
    b, a = butter_lowpass(cutoff, fs, order=order)

    # option 1: array
    if type(data) == np.ndarray:
        numeric_data = pd.DataFrame(data)
        y = numeric_data.apply(lambda col: filtfilt(b, a, col), axis=0)

    # option 2: dataframe
    else:
        # check if time 
        if 'time' in data.columns:
            numeric_data = data.select_dtypes(include=float).drop('time', axis=1)
        else:
            numeric_data = data.select_dtypes(include=float)
        
        # apply filter column-wise to numeric columns (v0.1.3); time dropped (v0.1.4)
        y = numeric_data.apply(lambda col: filtfilt(b, a, col), axis=0)
    
    return y

# compute q_dot, q_ddot for trial
def compute_q_dot_ddot(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = butter_lowpass_filter(data)

    # compute joint angular velocities (q dot)
    q_dot = diff_three_point(q)
    q_ddot = diff_three_point(q_dot)

    # add time, study_id columns
    q_dot.insert(0, 'time', data['time'])
    q_dot.insert(0, 'study_id', data['study_id'])
    q_ddot.insert(0, 'time', data['time'])
    q_ddot.insert(0, 'study_id', data['study_id'])

    return q_dot, q_ddot