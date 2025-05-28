import os
import io
import dropbox
import pandas as pd

# generate a list of all TRC files
    # folder_path: path to the folder containing TRC files
    # static: whether to include only static files
    # return_path: whether to return full path or just file name
def list_trc_files(dbx: dropbox.Dropbox, 
                   folder_path: str, 
                   static: bool = False,
                   return_path: bool = True):
    trc_files = []         # initial file list
    study_ids = []         # list of study IDs created from subject ID, file name
    has_more = True        # indicator for if there are more files in folder
    cursor = None          # cursor to point at files
    trc_folder = 'TRC'     # subfolder we're looking for

    # continue fetching entries until all are received
    while has_more:
        # check for cursor (None if first request)
        if cursor:
            result = dbx.files_list_folder_continue(cursor)
        else:
            result = dbx.files_list_folder(folder_path, recursive=True)

        # iterate through folder entries
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and f'/{trc_folder}/' in entry.path_display and entry.name.endswith('.trc'):
                # get subject ID from parent folder (this is really for creating study IDs)
                subject_id = os.path.basename(os.path.dirname(os.path.dirname(entry.path_display))) 

                if static:
                    if '@static' in entry.name:
                        # option 1: append full path
                        if return_path:
                            trc_files.append(entry.path_display)

                        # option 2: append short name (e.g., filename.trc)
                        else:
                            trc_files.append(entry.name)

                        # create study ID from subject ID and file name
                        study_id = f'{subject_id}_{entry.name.split("_")[-1]}'

                        # append study ID to list
                        study_ids.append(study_id.replace('.trc', ''))

                else:
                    # option 1: append full path
                    if return_path:
                        trc_files.append(entry.path_display)
                    
                    # option 2: append short name (e.g., filename.trc)
                    else:
                        trc_files.append(entry.name)

                    # create study ID from subject ID and file name
                    study_id = f'{subject_id}_{entry.name.split("_")[-1]}'

                    # append study ID to list
                    study_ids.append(study_id.replace('.trc', ''))
        
        # update handlers
        cursor = result.cursor
        has_more = result.has_more

    return study_ids, trc_files

# read TRC files into a DataFrame
def read_trc_file(dbx: dropbox.Dropbox, 
                  file_path: str,
                  return_marker_names: bool = False):
    # download the file as bytes
    _, response = dbx.files_download(path=file_path)
    file_content = response.content
    
    # convert bytes to a string
    trc_data = io.StringIO(file_content.decode('utf-8'))

    # read metadata from the first 4 lines
    metadata, marker_names = parse_trc_metadata(trc_data)

    # read TRC data into dataframe
    trc_df = pd.read_csv(trc_data, sep='\s+', on_bad_lines='skip')                     
    
    # reset index to track time column
    if isinstance(trc_df.index, pd.MultiIndex):
        trc_df_clean = trc_df.reset_index(level=1).rename(columns={'level_1': 'time'})
    else:
        trc_df_clean = trc_df.reset_index(drop=True)    # no multi-index; just reset to default integer index
    
    # check for properly formatted TRC file
    trc_df_formatted = check_trc_format(trc_df_clean)
    
    # create metadata dataframe
    metadata_df = create_metadata_df(metadata)
    if return_marker_names:
        return trc_df_formatted, metadata_df, marker_names
    
    return trc_df_formatted, metadata_df

# check for properly formatted TRC file
    # in some cases, `index` and `time` are saved as X1, Y1
    # so, we check for monotonically increasing X1; if so, we shift all columns two to the left
def check_trc_format(df: pd.DataFrame):
    # check for monotonic increasing X1 column
    if df['X1'].is_monotonic_increasing:
        time = df['Y1']                                             # save time values
        df_formatted = df.shift(-2, axis=1, fill_value=None)        # shift all columns two to the left
        df_formatted.insert(0, 'time', time)                        # insert time column at the beginning
    
        return df_formatted
    
    return df           # return original dataframe if no formatting needed

# get metadata (frames, markers, etc.) from TRC file
def parse_trc_metadata(trc_data: io.StringIO):
    # initialize variables for metadata
    metadata = {}
    
    # process each line to capture relevant metadata
    for _ in range(4):
        metadata_line = next(trc_data).strip()
        metadata_values = metadata_line.split('\t')
        
        # skip header labels (like "DataRate", "CameraRate")
        if 'DataRate' in metadata_values or 'CameraRate' in metadata_values:
            continue  # Skip this line and move to the next
        
        # parse numeric metadata
        if len(metadata_values) >= 5:  # Ensure there are enough columns
            try:
                # Extract relevant metadata
                metadata['frame_rate'] = float(metadata_values[0])
                metadata['num_frames'] = int(metadata_values[2])
                metadata['num_markers'] = int(metadata_values[3])
                metadata['units'] = metadata_values[4]
                break  # Stop after capturing relevant metadata
            except ValueError:
                continue  # Handle cases where conversion fails and skip the line
    
    # read the next line (marker names)
    marker_line = next(trc_data).strip()
    marker_names = marker_line.split('\t')

    # create marker mapping from names
    marker_mapping = create_marker_mapping(marker_names)

    return metadata, marker_mapping

# create a mapping of marker indices to marker names
def create_marker_mapping(marker_names: list):
    cleaned_marker_names = [name for name in marker_names if name.strip()]                      # remove empty strings
    marker_mapping = {index + 1: name for index, name in enumerate(cleaned_marker_names)}       # create mapping
    
    return marker_mapping

# create metadata df from dictionary
def create_metadata_df(metadata: dict):
    metadata_df = pd.DataFrame(list(metadata.items()), columns=['Metadata', 'Value']).T
    
    metadata_df.columns = metadata_df.iloc[0]                   # set the first row as the column headers
    metadata_df = metadata_df.drop(metadata_df.index[0])        # drop the first row
    metadata_df.reset_index(drop=True, inplace=True)            # reset the index

    return metadata_df