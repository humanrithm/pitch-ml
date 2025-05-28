import os, json
import pandas as pd
import opensim as osim
from typing import Union
from connections import AWS
from .xml import update_id_setup_xml

__version__ = '0.2.3'

# get subject info file from json
def get_subject_info(
        subject_id: str,
        path_stem: str = '../../../data/subjects',
) -> Union[dict | None]:
    # check if file exists
    if not os.path.exists(f'{path_stem}/{subject_id}/info.json'):
        return None
    
    with open(f'{path_stem}/{subject_id}/info.json', 'r') as f:
        subject_info = json.load(f)

    return subject_info

# create ID tool for a trial
        # updates a subject's ID setup file with trial-specific info
        # returns the ID tool and XML path
def create_id_tool(
        aws_connection: AWS,
        subject_id: str,
        trial_id: str,
        xml_file_name: str = 'id_setup.xml',
) -> osim.InverseKinematicsTool:
    # update setup file
    subject_info = get_subject_info(subject_id)
    brt = aws_connection.run_query(f'SELECT release_time FROM biomech.ball_release_frames WHERE "study_id" = \'{trial_id}\'').values[0][0]

    # update & load setup file
    xml_tree = update_id_setup_xml(trial_id, brt, subject_info['throws'])
    xml_tree.write(xml_file_name)
    
    # create IK tool
    id_tool = osim.InverseDynamicsTool(xml_file_name) 
    
    return id_tool

# get arm joint moments
def list_arm_moments() -> list:
    return [
       'arm_flex_r_moment', 'arm_add_r_moment', 'arm_rot_r_moment', 
       'arm_flex_l_moment', 'arm_add_l_moment', 'arm_rot_l_moment',
       'elbow_flex_r_moment', 'elbow_add_r_moment', 'elbow_pro_r_moment',
       'elbow_flex_l_moment', 'elbow_add_l_moment', 'elbow_pro_l_moment',
       'pro_sup_r_moment', 'pro_sup_l_moment', 
       'wrist_flex_r_moment', 'wrist_dev_r_moment',
       'wrist_flex_l_moment', 'wrist_dev_l_moment'
    ]

# get ID results as dataframe
def process_id_results(
        trial_id: str,
        subject_info: dict,
        mass_percent: float = 0.05
) -> pd.DataFrame:
    # read .sto file & insert study ID
    id_results = pd.read_csv(f'results/{trial_id}_id.sto', sep='\t', skiprows=6)
    id_results.insert(0, 'study_id', trial_id)

    # normalize using subject info
    for moment in list_arm_moments():
        if moment in id_results.columns:
            id_results[moment] = id_results[moment] / (subject_info['mass'] * mass_percent)

    # remove _l or _r from ID column names
    match subject_info['throws']:
        case 'right':
            id_results.columns = [col.replace('_r_', '_') for col in id_results.columns]
        case 'left':
            id_results.columns = [col.replace('_l_', '_') for col in id_results.columns]

    return id_results