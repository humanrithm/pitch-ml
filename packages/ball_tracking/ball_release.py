import pandas as pd

__version__ = '0.1.2'

# identify ball release (frame and time) from marker data
    # method: 'kinatrax' (Nebel 2024) or 'escamilla' (Escamilla 1998)
def identify_ball_release(
        data: pd.DataFrame,
        throwing_hand: str,
        release_buffer: int = 5,
        method: str = 'kinatrax'
) -> pd.DataFrame:
    
    if method == 'kinatrax':
        # use throwing hand to get release marker
        match throwing_hand:
            case "right":
                release_marker = "X18"
            case "left":
                release_marker = "X25"
        
        # get point of peak velocity of release marker, then add release buffer
        release_frame = data[release_marker].diff().argmax() + release_buffer
        release_time = data.loc[release_frame, "time"]

        # create dataframe with info
        release_df = pd.DataFrame((release_frame, release_time)).T
        release_df.columns = ['release_frame', 'release_time']

        return release_df
    
    elif method == 'escamilla':
        match throwing_hand:
            case "right":
                hand_marker = "X18"
                wrist_marker = "X16"
            case "left":
                hand_marker = "X25"
                wrist_marker = "X23"
        
        # get fist point at which wrist passes hand, then add buffer (Escamilla, Fleisig, Barrentine 1998)
        release_frame = data[data[wrist_marker] > data[hand_marker]].index[0] + release_buffer
        release_time = data.loc[release_frame, "time"]

        # create dataframe with info
        release_df = pd.DataFrame((data['study_id'].values[0], release_frame, release_time)).T
        release_df.columns = ['study_id', 'release_frame', 'release_time']

        return release_df