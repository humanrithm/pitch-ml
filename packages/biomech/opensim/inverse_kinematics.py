import pandas as pd

__version__ = '0.1.0'

# load a .mot file (e.g., results from an IK run)
def load_mot_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delim_whitespace=True, skiprows=10)

# create dataframe w/ IK errors
def process_ik_errors(path: str = '_ik_marker_errors.sto',) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(path, sep='\t', skiprows=5, header=1)