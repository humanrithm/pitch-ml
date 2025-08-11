import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

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
        noninj[matching_cols],
        inj[matching_cols],
        metric=metric
    ).flatten()

    # get min. distance
    min_idx = distances.argmin()
    min_distance = distances[min_idx]
    matched_pitcher_id = noninj.iloc[min_idx]['mlbamid']

    return {
        'mlbamid_injured': inj['mlbamid'].values[0],
        'mlbamid_noninjured': matched_pitcher_id,
        'min_distance': min_distance
    }