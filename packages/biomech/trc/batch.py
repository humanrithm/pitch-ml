import dropbox
import pandas as pd
from .load import read_trc_file, list_trc_files

# return list of all subjects for iterating
def get_all_subjects(dbx: dropbox.Dropbox):
    subject_folders = dbx.files_list_folder('/PhD/Data/MOTUS/PRO/').entries
    subjects = [entry.name for entry in subject_folders if isinstance(entry, dropbox.files.FolderMetadata)]
    
    return subjects

# get all TRC files for a subject
    # dbx: Dropbox object
    # subject_id: subject ID
    # static: whether to include only static files
def get_subject_trc_files(dbx: dropbox.Dropbox, 
                          subject_id: str,
                          static: bool):
    folder_path = f'/PhD/Data/MOTUS/PRO/{subject_id}'
    study_ids, trc_files = list_trc_files(dbx, folder_path, static)

    print(f'Found {len(trc_files)} TRC files for subject {subject_id}', end='\r', flush=True)
    
    return study_ids, trc_files

# read a subject's TRC files
def read_subject_trc_files(dbx: dropbox.Dropbox,
                           study_ids: list,
                           trc_files: list):
    # initialize full lists of data
    full_trc_data = []
    full_metadata = []

    # loop through all TRC files
    for study_id, trc_file in zip(study_ids, trc_files):
        try:
            # read the TRC file
            trc_df, metadata = read_trc_file(dbx, trc_file)
            
            # insert study ID into dataframes
            trc_df.insert(0, 'study_id', study_id)
            metadata.insert(0, 'study_id', study_id)
            
            # append to full lists if sufficient data
            if trc_df.shape[0] > 500:
                full_trc_data.append(trc_df)
                full_metadata.append(metadata)

        except Exception as e:
            print(f'Error processing {trc_file}: {e}', flush=True)

    if len(full_trc_data) == 0:
        return None, None
    
    return pd.concat(full_trc_data), pd.concat(full_metadata)

# batch process subjects
    # dbx: Dropbox object
    # batch: list of subjects to process
    # static: whether to include only static files
def batch_process(dbx: dropbox.Dropbox,
                  batch: list,
                  static: bool):
    # set batch name 
    batch_name = f'trc_raw_{batch[0]}_{batch[-1]}'
    
    # initialize full lists of data
    full_trc_data = []
    full_metadata = []

    # loop through all subjects
    for subject in batch:
        # get & read TRC files for subject
        study_ids, trc_files = get_subject_trc_files(dbx, subject, static)
        trc_data, metadata = read_subject_trc_files(dbx, study_ids, trc_files)

        # skip if no TRC data
        if trc_data is None:
            print(f'No TRC data found for subject {subject}', flush=True)
            continue
        
        # check for extra columns (subject_id, time, + markers)
        if len(trc_data.columns) > 140:
            trc_data = trc_data.iloc[:, 0:140]
        
        # append to full lists
        full_trc_data.append(trc_data)
        full_metadata.append(metadata)

        print(f'Processed all {len(trc_files)} TRC files for subject {subject}', flush=True)
    
    return pd.concat(full_trc_data), pd.concat(full_metadata), batch_name

# clean up TRC data
def clean_trc_data(data: pd.DataFrame):
    copy = data.copy()

    # trim to markered columns
    copy = copy.iloc[:, 0:46]