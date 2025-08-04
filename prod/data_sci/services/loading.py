import pandas as pd

# load a .mot file (e.g., results from an IK run)
def load_mot_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delim_whitespace=True, skiprows=10)

# create normalized time column
def compute_normalized_time(
        data: pd.DataFrame
) -> pd.DataFrame:
    # create normalized time
    if 'normalized_time' not in data.columns:
        data.insert(
            2, 
            'normalized_time', 
            data.groupby('study_id')['time'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )

    return data