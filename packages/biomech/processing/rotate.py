import numpy as np
import pandas as pd

def rotate_data(
        data: pd.DataFrame,
        rotation_matrix: np.ndarray
) -> pd.DataFrame:
    # create copy of data to avoid overwriting
    copy = data.copy()  

    # apply rotation matrix to each marker
    for i in range(1, int(len(data.columns) / 3) + 1):
        # get rotated marker data
        marker_vector = copy[[f"X{i}", f"Y{i}", f"Z{i}"]].values
        rotated_marker = np.dot(rotation_matrix, marker_vector.T).T  

        # update dataframe with rotated marker data
        copy[f"X{i}"] = rotated_marker[:, 0]
        copy[f"Y{i}"] = rotated_marker[:, 1]
        copy[f"Z{i}"] = rotated_marker[:, 2]

    return copy