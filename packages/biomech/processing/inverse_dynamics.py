import os
import pandas as pd

# get all .mot files from IK 
    # requires path to subjects directory relative to current working directory
def get_mot_files(subjects_dir: str = '../../../data/subjects/') -> list:
    mot_files = []
    for subject in os.listdir(subjects_dir):
        results_dir = os.path.join(subjects_dir, subject, 'results')
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.endswith('.mot'):
                    mot_files.append(os.path.join(results_dir, file))
    return mot_files

# read .mot file from IK
def read_mot_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='\t', skiprows=10)
        