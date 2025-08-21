import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# compute Euclidean distance between injured pitcher and all eligible non-injured pitchers
def compute_matching_info(
        inj: pd.DataFrame,
        noninj: pd.DataFrame,
        matching_cols: list,
        metric: str = 'euclidean',
        n_matches: int = 1
) -> dict:
    """ Computes the distance between injured pitcher and all eligible non-injured pitchers. Returns a dictionary with distances and corresponding non-injured pitcher IDs. """

    # compute distance btw injured pitcher & all eligible non-injured pitchers based on matching cols
    distances = cdist(
        noninj[matching_cols],
        inj[matching_cols],
        metric=metric
    ).flatten()

    # get min. distance(s)
    if n_matches > 1:
        sorted_indices = distances.argsort()[:n_matches]
        matched_pitcher_ids = noninj.iloc[sorted_indices]['mlbamid'].tolist()
        min_distances = distances[sorted_indices].tolist()
        return {
            'mlbamid_injured': inj['mlbamid'].values[0],
            'mlbamid_noninjured': matched_pitcher_ids,
            'min_distances': min_distances
        }
    else:
        min_idx = distances.argmin()
        min_distance = distances[min_idx]
        matched_pitcher_id = noninj.iloc[min_idx]['mlbamid']

    return {
        'mlbamid_injured': inj['mlbamid'].values[0],
        'mlbamid_noninjured': matched_pitcher_id,
        'min_distance': min_distance
    }