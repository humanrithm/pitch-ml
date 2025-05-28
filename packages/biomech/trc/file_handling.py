import os
import numpy as np
import pandas as pd 
from typing import Union
from biomech.processing import list_markers

__version__ = '0.1.7'

# create trc header from dataframe info
def create_trc_header(
        subject_id: str,
        data: pd.DataFrame,
        throwing_hand: str,
        frame_rate: int = 480
) -> list:
    
    # get markers in model
    markers = list_markers(throwing_hand)
    num_markers = len(markers)
    marker_names = "\t\t\t".join([f"{m}" for m in markers])     # still need to add three tabs after last
    marker_axes = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)])
    
    return [
        f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(f'{subject_id}')}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{frame_rate}\t{frame_rate}\t{len(data)}\t{num_markers}\tmm\t{frame_rate}\t1\t{len(data)}",
        f"Frame#\tTime\t{marker_names}\t\t\t",
        f"\t\t{marker_axes}\t\n",
    ]

# create trc body
def create_trc_body(
        data: pd.DataFrame
) -> pd.DataFrame:
    if isinstance(data, str):
        trc_body = pd.read_csv(data)
    else:
        trc_body = data.copy()
    
    # 1-indexed (so Frame# starts at 1)
    trc_body.index = trc_body.index + 1 

    # drop columns that don't fit TRC format (eg., study_id)
    if 'study_id' in trc_body.columns:
        trc_body.drop('study_id', axis=1, inplace=True)  

    return trc_body

def rotate_trc_body(
        data: pd.DataFrame,
        rotation_matrix: np.ndarray,
        throwing_hand: str
) -> pd.DataFrame:
    # create copy of data to avoid overwriting
    copy = data.copy()  
    
    # get markers in model
    marker_list = list_markers(throwing_hand)

    # apply rotation matrix to each marker
    for marker in marker_list:
        # get rotated marker data
        marker_vector = copy[[f"{marker}X", f"{marker}Y", f"{marker}Z"]].values
        rotated_marker = np.dot(rotation_matrix, marker_vector.T).T  

        # update dataframe with rotated marker data
        copy[f"{marker}X"] = rotated_marker[:, 0]
        copy[f"{marker}Y"] = rotated_marker[:, 1]
        copy[f"{marker}Z"] = rotated_marker[:, 2]

    return copy

# create TRC objects (head, body) from dataframe
    # subject: subject_id (for TRC header)
    # data: dataframe w/ marker data OR csv file path (for TRC body)
def create_trc_objects(
        subject_id: str,
        data: Union[str | pd.DataFrame],
        throwing_hand: str,
        rotation_matrix: np.ndarray = None 
) -> tuple[list[str], str]:

    # create TRC body (rotate if necessary)
    trc_body = create_trc_body(data)
    if rotation_matrix is not None:
        trc_body = rotate_trc_body(trc_body, rotation_matrix, throwing_hand)
    
    # create TRC header
    trc_header = create_trc_header(subject_id, trc_body, throwing_hand)
    
    return trc_header, trc_body

# write to TRC file
def write_to_trc(
        file_path: str,
        header: list,
        body: pd.DataFrame
) -> None:
    with open(file_path, 'w') as f:
        f.write("\n".join(header) + "\n")
        body.to_csv(f, sep="\t", index=True, header=False)