import io
import os
import re
import dropbox
import pandas as pd

# download & read CSV files
def download_csv_files(dbx: dropbox,
                       csv_files: list,
                       device: str):
    data_list = []
    for file_path in csv_files:
        try:
            # download file as bytes
            _, response = dbx.files_download(path=file_path)
            file_content = response.content

            # convert bytes to a file-like object and read into a df 
            if device == 'trackman':
                raw_df = pd.read_csv(io.BytesIO(file_content), header=2, usecols=lambda col: col != 0, encoding='utf-8')
            if device == 'rapsodo':
                raw_df = pd.read_csv(io.BytesIO(file_content), header=2, encoding='utf-8')

            # clean data based on device (find header row, add subject ID)
            clean_df = clean_csv_data(raw_df, device, file_path)
            clean_df_with_ids = create_study_ids(file_path, clean_df, device)

            # append to full dataframe
            data_list.append(clean_df_with_ids)

        except dropbox.exceptions.ApiError as err:
            print(f"Error downloading {file_path}: {err}")
            continue  # skip to the next file in case of error

    return pd.concat(data_list).reset_index(drop=True)

def clean_csv_data(data: pd.DataFrame,
                   device: str,
                   file_path: str):
    try:
        # check for appropriate header & correct if necessary 
        # data = check_header(data)

        # drop duplicate columns (will end in .*)
        data = drop_duplicate_columns(data)
        
        # device-specific adjustments
            # TM: drop identifiers, hitter columns
            # Rapsodo: drop identifiers
        match device: 
            case 'trackman':
                # remove pitcher info
                pitcher_cols = ['Pitcher', 'PitcherTeam']
                data.drop(columns=[col for col in pitcher_cols if col in data.columns], axis=1, inplace=True)            
                
                # drop hitter columns
                hitter_cols = ['Direction', 'BatterId', 'Batter', 'HitSpinRate', 'HitType',
                            'ExitSpeed', 'BatterSide', 'Angle', 'PositionAt110X', 'PositionAt110Y',
                            'PositionAt110Z', 'Distance', 'LastTrackedDistance', 'HangTime',
                            'Bearing', 'ContactPositionX', 'ContactPositionY', 'ContactPositionZ']
                data.drop(hitter_cols, axis=1, inplace=True)
                
                # drop redundant identifier
                if 'MAC' in data.columns:
                    data.drop('MAC', axis=1, inplace=True) 

                if 'test' in data.columns:
                    data.drop('MAC', axis=1, inplace=True) 

            case 'rapsodo': 
                if 'MAC' in data.columns:
                    data.drop('MAC', axis=1, inplace=True) 

                if 'test' in data.columns:
                    data.drop('MAC', axis=1, inplace=True)

        return data
    
    except Exception as err:
        print(f"Error cleaning data in {file_path}: {err}")
        
        return data

# check for appropriate header 
def check_header(data: pd.DataFrame):   
    try:
        pattern = re.compile(r'Unnamed: \d+')                          # pattern matching for 'Unnamed: [digit]' in columns
        
        # decode column names if they are in bytes
        data.columns = [str(col) if isinstance(col, bytes) else col for col in data.columns]
        
        while any(pattern.match(col) for col in data.columns):
            data.columns = list(data.loc[0, :].values)                 # set columns equal to first row
            data = data.loc[1:, :]                                     # remove first row

        return data
    
    except Exception as err:
        print(f"Error checking header: {err}")
        return data

# drop duplicate columns
def drop_duplicate_columns(data: pd.DataFrame):
        pattern = re.compile(r'.*\.\w+$')                                           # any column ending in .*
        matching_columns = [col for col in data.columns if pattern.match(col)]      # get matching columns
        data.drop(columns=matching_columns, inplace=True)                           # drop matches from data

        return data

# create study ID ([subject-no]_[pitch-no])
def create_study_ids(file_path: str,
                     data: pd.DataFrame,
                     device: str):
    try:
        subject_id = os.path.dirname(file_path).split('/')[-1]                  # get subject ID from file path

        # create IDs based on device
        if device == 'trackman':
            study_ids = [f'{subject_id}_{pitch_num}' for pitch_num in data['PitchNo']]   # create identifiers
        elif device == 'rapsodo':
            study_ids = [f'{subject_id}_{pitch_num}' for pitch_num in data['No']]   # create identifiers

        # insert IDs into data
        data.insert(0, 'studyID', study_ids)

        return data
    
    except Exception as err:
        print(f"Error creating study IDs in {file_path}: {err}")
        return data