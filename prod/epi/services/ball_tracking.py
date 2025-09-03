import pandas as pd

def clean_ball_tracking_data(
        data: pd.DataFrame,
        model_fts: list = ['rel_speed', 'rel_side', 'rel_ht', 'spin_rate', 'spin_axis', 'ax0', 'ay0', 'az0']
) -> pd.DataFrame:
    """
    Cleans the ball tracking data by trimming to necessary columns, removing outliers, etc.
    """
    data.rename(columns={
        'release_speed': 'rel_speed',
        'release_pos_x': 'rel_side',
        'release_pos_z': 'rel_ht',
        'release_spin_rate': 'spin_rate',
        'spin_axis': 'spin_axis',
        'ax': 'ax0',
        'ay': 'ay0',
        'az': 'az0',
    }, inplace=True)

    # setup model dataset for cohort
    clean_data = data[['pitcher', 'p_throws', 'game_date', 'pitcher_days_since_prev_game', 'pitch_type'] + model_fts].copy().reset_index(names='pitch_id')
    
    return clean_data

# get list of seasons from a date column (e.g., game_date)
    # used to determine season(s) from which a pitcher has pitched
def get_season_from_date(date: pd.Series) -> list:
    """
    Extracts the list of season(s) from a date column.
    """
    date_dt = pd.to_datetime(date, errors='coerce')
    seasons = date_dt.apply(lambda x: x.year if pd.notnull(x) else None)

    return list(seasons.dropna().unique())

def pivot_pitch_labels(
        data: pd.DataFrame.applymap,
        pitcher_id_col: str = 'pitcher',
        pitch_id_col: str = 'pitch_id',
) -> pd.DataFrame:
    """ Aggregate and pivot pitch labels into wide format. """
    counts = data.groupby([pitcher_id_col, 'season', 'pitch_type'])[pitch_id_col].count().reset_index().rename(columns={'pitch_id': 'pitches_thrown'})
    counts_pivot = counts.pivot_table(
        index=['pitcher', 'season'],
        columns='pitch_type',
        values='pitches_thrown',
        fill_value=0
    )
    counts_pivot.columns = [f"{pt}_count" for pt in counts_pivot.columns]
    counts_pivot = counts_pivot.reset_index()

    return counts_pivot