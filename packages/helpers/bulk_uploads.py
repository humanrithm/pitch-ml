import os 
import pandas as pd

# load all errors
def load_error_files(folder: str = 'errors') -> pd.DataFrame:
    files = os.listdir(folder)
    errors = []
    for file in files:
        error_df = pd.read_csv(f'{folder}/{file}')  
        
        # add study ID if not present
        if 'study_id' not in error_df.columns:
            error_df.insert(0, 'study_id', file.split('.')[0])
        
        errors.append(error_df)
    
    return pd.concat(errors)

# load all joint angles
def load_joint_angles(folder: str = 'joint_angles') -> pd.DataFrame:
    # get files
    files = [f for f in os.listdir(folder) if not f.__contains__('_br')]    
    joint_angles = []
    
    # iterate through files
    for file in files:
        joint_angle_df = pd.read_csv(f'{folder}/{file}')  
        
        # add study id if not present
        if 'study_id' not in joint_angle_df.columns:
            joint_angle_df.insert(0, 'study_id', file.split('.')[0])
        
        joint_angles.append(joint_angle_df)
    
    return pd.concat(joint_angles).sort_values(by=['study_id', 'time'])

def create_file_paths(
        subject_id: str, 
        ik_path_stem: str = '../../bulk/results/elbow_pin_joint/inverse_kinematics',
) -> dict:
    return {
        'joint_angles': f'{ik_path_stem}/joint_angles/full/{subject_id}.csv',
        'errors': f'{ik_path_stem}/errors/{subject_id}.csv'
    }

# load joint torques
def load_joint_torques(folder: str = 'joint_torques') -> dict:
    # get all files (full, br, and max valgus)
    files = {
        'full': [f for f in os.listdir(folder) if f.__contains__('_full')],
        'ball_release': [f for f in os.listdir(folder) if f.__contains__('_br')],
        'max_valgus': [f for f in os.listdir(folder) if f.__contains__('_max_valgus')]
    }
    
    # results storage
    joint_torques = {
        'full': [],
        'ball_release': [],
        'max_valgus': []
    }
    
    for file_type, file_list in files.items():
        for f in file_list:
            joint_torque_df = pd.read_csv(f'{folder}/{f}')  
            joint_torques[file_type].append(joint_torque_df)

        joint_torques[file_type] = pd.concat(joint_torques[file_type])
    
    return joint_torques