import pandas as pd
from scipy.spatial.distance import cdist

# compute Euclidean distance between injured pitcher and all eligible non-injured pitchers
def compute_matching_info(
        inj: pd.DataFrame,
        noninj: pd.DataFrame,
        matching_cols: list,
        metric: str = 'euclidean'
) -> dict:
    """ Computes the distance between injured pitcher and all eligible non-injured pitchers. Returns a dictionary with distances and corresponding non-injured pitcher IDs. """
    # compute distance btw injured pitcher & all eligible non-injured pitchers based on matching cols
    distances = cdist(
        noninj[matching_cols].values,
        inj[matching_cols].values,
        metric=metric
    ).flatten()

    # get min. distance
    min_idx = distances.argmin()
    min_distance = distances[min_idx]
    matched_pitcher_id = noninj.iloc[min_idx]['mlbamid']

    return {
        'injured_id': inj['mlbamid'].values[0],
        'matched_id': matched_pitcher_id,
        'min_distance': min_distance,
        'inj_mass': inj['mass'].values[0],
        'inj_height': inj['height'].values[0],
        'inj_pitches_thrown': inj['pitches_thrown_interval'].values[0],
        'noninj_mass': noninj.iloc[min_idx]['mass'],
        'noninj_height': noninj.iloc[min_idx]['height'],
        'noninj_pitches_thrown': noninj.iloc[min_idx]['pitches_thrown_interval'],
    }