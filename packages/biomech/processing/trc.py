import io, os
import pandas as pd
from typing import Union, TextIO
from biomech.processing import create_marker_mapping, list_markers

__version__ = '0.1.0'

# lists of markers to preserve in TRCs
__markers_left__ = [
    'X19', 'Y19', 'Z19', # 19,left_bicep
    'X20', 'Y20', 'Z20', # 20,left_lateral_elbow
    'X21', 'Y21', 'Z21', # 21,left_medial_elbow
    'X22', 'Y22', 'Z22', # 22,left_forearm
    'X23', 'Y23', 'Z23', # 23,left_lateral_wrist
    'X24', 'Y24', 'Z24', # 24,left_medial_wrist
    'X25', 'Y25', 'Z25'  # 25,left_hand
]
__markers_right__ = [
    'X12', 'Y12', 'Z12', # 12,right_bicep
    'X13', 'Y13', 'Z13', # 13,right_lateral_elbow
    'X14', 'Y14', 'Z14', # 14,right_medial_elbow
    'X15', 'Y15', 'Z15', # 15,right_forearm
    'X16', 'Y16', 'Z16', # 16,right_lateral_wrist
    'X17', 'Y17', 'Z17', # 17,right_medial_wrist
    'X18', 'Y18', 'Z18'  # 18,right_hand
]


""" TRC COMPILATION """

# write to TRC file
def write_to_trc(
        file_name: str,
        body: pd.DataFrame, 
        throwing_hand: str,
        frame_rate: int = 480
) -> None:
    """ Write a TRC file with the given header and body data. Uses `throwing_hand` to determine which markers to include. 
    
    **Note**: Assumes markers are already rotated to match desired coordinate system. """
    
    # create header
    header = create_trc_header(file_name, body, throwing_hand, frame_rate)

    # filter body to include only markers in the model
    if throwing_hand == 'left':
        body = body[['Frame#', 'Time'] + __markers_left__]
    elif throwing_hand == 'right':
        body = body[['Frame#', 'Time'] + __markers_right__]
    else:
        raise ValueError("Invalid throwing hand specified. Use 'left' or 'right'.")
    
    # write to file
    with open(file_name, 'w') as f:
        f.write("\n".join(header) + "\n")
        body.to_csv(f, sep="\t", index=False, header=False)

""" TRC HEADER PROCESSING """
# create trc header from dataframe info
def create_trc_header(
        file_name: str,
        body: pd.DataFrame,
        throwing_hand: str,
        frame_rate: int
) -> list:
    
    """ Create a TRC header for the given subject and data.
    
    **Args:**
        **subject_id** (str): Subject identifier.
        **data** (pd.DataFrame): Data containing marker positions.
        **throwing_hand** (str): The throwing hand of the subject. Used to pull the necessary markers.
        **frame_rate** (int): Frame rate of the data. Default is 480."""
    
    # get markers in model
    markers = list_markers(throwing_hand)
    num_markers = len(markers)
    marker_names = "\t\t\t".join([f"{m}" for m in markers])     # still need to add three tabs after last
    marker_axes = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)])
    
    return [
        f"PathFileType\t4\t(X/Y/Z)\t{file_name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{frame_rate}\t{frame_rate}\t{len(body)}\t{num_markers}\tmm\t{frame_rate}\t1\t{len(body)}",
        f"Frame#\tTime\t{marker_names}\t\t\t",
        f"\t\t{marker_axes}\t\n",
    ]

""" TRC BODY PROCESSING """
def parse_trc_body(
        source: Union[str, bytes, TextIO],
        sample_rate: int = 480
) -> pd.DataFrame:
    
    """Parse the body of a TRC file or string into a DataFrame. Returns a DataFrame with all marker positions."""

    # create a TextIO stream from the source
    if isinstance(source, bytes):
        trc_stream = io.StringIO(source.decode('utf-8'))
    elif isinstance(source, str):
        trc_stream = io.StringIO(source)
    else:
        trc_stream = source
    
    # read stream as CSV, skipping the first 4 lines (header)
    trc_data = pd.read_csv(trc_stream, skiprows=4, sep='\s+', on_bad_lines='skip')

    # handle first two columns (Frame# and Time)
    if isinstance(trc_data.index, pd.MultiIndex):
        trc_data_clean = trc_data.reset_index(level=1).rename(columns={'level_1': 'Time'})
    elif 'Time' not in trc_data.columns:
        trc_data_clean = trc_data.reset_index(drop=True)                              # no multi-index; just reset to default integer index
        trc_data_clean.insert(0, 'Time', trc_data_clean.index * (1 / sample_rate))    # insert time column based on index

    # insert frame number
    trc_data_clean.insert(0, 'Frame#', range(1, len(trc_data_clean) + 1))

    return trc_data_clean
