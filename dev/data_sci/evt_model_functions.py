import pandas as pd

# mirror columns to match RHP
def mirror_columns(
        data: pd.DataFrame,
        cols: list
):
    data_mirrored = data.copy()
    for col in cols:
        data_mirrored.loc[data['throws'] == 'left', col] *= -1
    
    return data_mirrored