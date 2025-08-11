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
    clean_data = data[['pitcher', 'p_throws', 'game_date', 'pitcher_days_since_prev_game'] + model_fts].copy().reset_index(names='pitch_id')
    
    return clean_data