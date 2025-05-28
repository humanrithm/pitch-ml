import json, os
import pandas as pd
import opensim as osim
from typing import Union
from connections import AWS
from .xml import update_ik_setup_xml

__version__ = '0.2.6'

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

# create IK tool for a trial
        # updates a subject's IK setup file with trial-specific info
        # returns the IK tool and XML path
def create_ik_tool(
        aws_connection: AWS,
        subject_id: str,
        trial_id: str,
        xml_file_name: str = 'ik_setup.xml',
) -> osim.InverseKinematicsTool:
    # update setup file
    subject_info = get_subject_info(subject_id)
    brt = aws_connection.run_query(f'SELECT release_time FROM biomech.ball_release_frames WHERE "study_id" = \'{trial_id}\'').values[0][0]

    # update & load setup file
    xml_tree = update_ik_setup_xml(trial_id, brt, subject_info['throws'])
    xml_tree.write(xml_file_name)
    
    # create IK tool
    ik_tool = osim.InverseKinematicsTool(xml_file_name) 
    
    return ik_tool

# load results from an IK run
def load_ik_results(
        trial_id: str
) -> pd.DataFrame:
    mot_file_path = f'results/{trial_id}_ik.mot'
    mot_df = pd.read_csv(mot_file_path, delim_whitespace=True, skiprows=10)
    
    return mot_df

# get joint angles for throwing arm
def get_joint_angles(
        data: pd.DataFrame,
        throwing_hand: str,
        joint_cols = ['arm_flex', 'arm_add', 'arm_rot', 'elbow_flex', 'pro_sup', 'wrist_flex', 'wrist_dev']
) -> pd.DataFrame:
    # specify joint columns based on throwing hand
    match throwing_hand:
        case 'left':
            joint_cols = ['time'] + [joint + '_l' for joint in joint_cols] 
        case 'right':
            joint_cols = ['time'] + [joint + '_r' for joint in joint_cols] 

    # filter data to joint angle columns
    joint_angles = data[joint_cols]
    
    return joint_angles

# process IK results for storage
def process_ik_results(
        trial_id: str,
        throwing_hand: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get motion data
    mot_df = load_ik_results(trial_id) 

    # get joint angles // filter to ball release
    joint_angles = get_joint_angles(mot_df, throwing_hand)
    joint_angles.insert(0, 'study_id', trial_id)
    
    return joint_angles

# create dataframe w/ IK errors
def process_ik_errors(
        trial_id: str,
        path_stem: str = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # get .sto file path w/ errors
    if path_stem:
        file_path = f'{path_stem}/{trial_id.split("_")[0]}/_ik_marker_errors.sto'
    else:
        file_path = '_ik_marker_errors.sto'
    
    # load errors (overall & ball release)
    ik_errors = pd.read_csv(file_path, sep='\t', skiprows=5, header=1)
    ik_errors.insert(0, 'study_id', trial_id)

    return ik_errors