import re
import pandas as pd
from typing import Union

__version__ = '0.1.7'

# create a dictionary with marker numbers as keys and anatomical labels as values
def create_marker_dict(marker_labels: pd.DataFrame) -> dict:
    return pd.Series(marker_labels['marker_name'].values, index=marker_labels['marker_number']).to_dict()

# map marker labels (e.g., X1) to anatomical names (e.g., front_head_X)
def map_marker_labels(
        data: pd.DataFrame,
        marker_map: Union[pd.DataFrame | dict]
    ) -> pd.DataFrame:
    
    # get dictionary and setup renamed columns
    renamed_columns = {}
    if type(marker_map) == pd.DataFrame:
        marker_map = create_marker_dict(marker_map)     # convert to dictionary if DataFrame

    for col in data.columns:
        # use regex to extract the marker number for a given format (e.g., X1, Y1)
        match = re.match(r'([XYZ])(\d+)', col)
        
        if match:
            axis, marker_num = match.groups()
            
            # construct new column name with anatomical label if marker number is in the dictionary
            if int(marker_num) in marker_map:
                new_name = f"{marker_map[int(marker_num)]}{axis}"
                renamed_columns[col] = new_name
            else:
                renamed_columns[col] = col      # leave as is if no mapping found
        
        else:
            renamed_columns[col] = col          # leave as is if no match found

    return data.rename(columns=renamed_columns)

# function to preserve columns from markers in the model
def preserve_columns(
        data: pd.DataFrame,
        throwing_hand: str
) -> pd.DataFrame:
    required_cols = list_required_columns(throwing_hand)
    return data[required_cols]

# list of markers + nec. columns present in the scaled model
def list_required_columns(throwing_hand: str) -> list:
    markers_in_model = list_markers(throwing_hand)
    required_cols = ['time']
    
    # extend list with columns for each marker
    for marker in markers_in_model:
        required_cols.extend([f"{marker}X", f"{marker}Y", f"{marker}Z"])

    return required_cols

# list of markers in the model based on throwing hand
def list_markers(throwing_hand: str) -> list:
    # list of markers present in the scaled model
    if throwing_hand == 'right':
        markers_in_model = [
            "right_bicep", "right_lateral_elbow", "right_medial_elbow", "right_forearm",
            "right_lateral_wrist", "right_medial_wrist", "right_hand"
        ]

    elif throwing_hand == 'left':
        markers_in_model = [
            "left_bicep", "left_lateral_elbow", "left_medial_elbow", "left_forearm",
            "left_lateral_wrist", "left_medial_wrist", "left_hand"
        ]

    else:
        raise ValueError('Throwing hand not specified.')

    return markers_in_model
