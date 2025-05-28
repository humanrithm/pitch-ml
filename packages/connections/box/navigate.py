import io
import os
import dropbox
import pandas as pd
from dotenv import load_dotenv

# function to iterate through directory & sub-directory
    # tracker: one of 'trackman' or 'rapsodo'
def list_csv_files(dbx: dropbox,
                   folder_path: str,
                   tracker: str,
                   return_path: bool = True):
    csv_files = []          # initial file list
    subject_ids = []        # initial subject ID list
    has_more = True         # indicator for if there are more files in folder
    cursor = None           # cursor to point at files

    # match tracker to determine file(s) to search for
    match tracker:
        case 'trackman': 
            file_target = 'trackman.csv'
        case 'rapsodo': 
            file_target = 'rapsodo.csv'
        case _:
            file_target = '.csv'

    # continue fetching entries until all are received
    while has_more:
        # check for cursor (None if first request)
        if cursor:
            result = dbx.files_list_folder_continue(cursor)
        else:
            result = dbx.files_list_folder(folder_path, recursive=True)
        
        # iterate through folder entries (searching for target file)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith(file_target):
                subject_ids.append(os.path.basename(folder_path))          # append subject ID
                
                # option 1: append full path
                if return_path:
                    csv_files.append(entry.path_display)                    # append full path
                # option 2: append short name (e.g., rapsodo.csv)
                else:
                    csv_files.append(entry.name)                    # append full path
        
        # update handlers
        cursor = result.cursor
        has_more = result.has_more

    return subject_ids, csv_files

# method to move non-TM files using DBX API
def move_non_trackman_files(dbx: dropbox,
                            parent_folder: str,
                            folder_path: str,
                            tracker: str = None):
    try:
        # get files in the folder
        _, csv_files = list_csv_files(dbx, folder_path, tracker, return_path=False)

        # check if both TrackMan and Rapsodo are present
        if 'trackman.csv' in csv_files and 'rapsodo.csv' in csv_files:
            subject_number = os.path.basename(folder_path) 
            discarded_folder = f'{folder_path}/discarded_{subject_number}'  # create subject's discarded folder w/ subject number

            # check if the discarded folder already exists
            try:
                dbx.files_get_metadata(discarded_folder)
            except dropbox.exceptions.ApiError as err:
                if isinstance(err.error, dropbox.files.GetMetadataError):
                    # implies folder does not exist, so catch exception & create folder
                    dbx.files_create_folder_v2(discarded_folder)
                    print(f"Created folder {discarded_folder}.")
                else:
                    raise  # re-raise the exception if it's not a folder existence issue
            
            # iterate over list of CSV files
            for file_name in csv_files:
                # move non-trackman files to discarded folder
                if file_name != 'trackman.csv':
                    source_path = f'{folder_path}/{file_name}'
                    dest_path = f'{discarded_folder}'

                    try:
                        dbx.files_get_metadata(source_path)
                        print(f"File exists: {source_path}")

                        # attempt to move the file
                        dbx.files_move_v2(source_path, dest_path)
                        print(f"Moved {file_name} to {discarded_folder}.")
                    
                    except dropbox.exceptions.ApiError as err:
                        # option 1: file already moved
                        if isinstance(err.error, dropbox.files.GetMetadataError):
                            print(f"File {file_name} not found, it may have already been moved.")
                        # option 2: new error, raise
                        else:
                            raise err
                            print(source_path, dest_path)

            # last: move the subject's discarded folder to the parent `Discarded Ball Tracking Data` folder
            landing_folder = f'{parent_folder}/Discarded Ball Tracking Data/discarded_{subject_number}'
            try:
                dbx.files_move_v2(discarded_folder, landing_folder)
                print(f'Moved {discarded_folder} to {landing_folder}.')
            
            except dropbox.exceptions.ApiError as move_err:
                # option 1: folder already moved
                if isinstance(err.error, dropbox.files.GetMetadataError):
                    print(f"Folder {file_name} not found, it may have already been moved.")
                # option 2: new error, raise
                else:
                    raise move_err

            print(f'Non-TrackMan files discarded and moved for {subject_number}.')

    except dropbox.exceptions.ApiError as err:
        print(f"Error processing folder {folder_path}: {err}")
