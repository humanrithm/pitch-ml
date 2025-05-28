import xml.etree.ElementTree as ET

__version__ = '0.2.4' # v0.2.3 -- updated FC -> BR window

# create file paths for ik setup xml
    # NOTE: this no longer uses subject_id because ik_run has been moved to the subject's folder (to make paths work)
def create_ik_file_paths(
        trial_id: str) -> tuple[str, str, str]:
    # create file paths
    scaled_model_path = f'scaled_model.osim'
    input_trc_file = f'trials/{trial_id}.trc'
    output_motion_file = f'results/{trial_id}_ik.mot'

    return scaled_model_path, input_trc_file, output_motion_file

# update ik setup xml file
def update_ik_setup_xml(
    trial_id: str,
    ball_release_time: float,
    throwing_hand: str,
    time_offset: float = 0.217,                         # NOTE: defined based on conservative window from Escamilla, Fleisig (1998)
    xml_path_stem: str = '../../xml_templates'
) -> None:
    # grab & parse correct template file based on throwing hand
    tree = ET.parse(f'{xml_path_stem}/ik_{throwing_hand}.xml')
    root = tree.getroot()

    # create file paths
    file_paths = create_ik_file_paths(trial_id)

    # update file paths
    root.find('.//model_file').text = file_paths[0]
    root.find('.//marker_file').text = file_paths[1]
    root.find('.//output_motion_file').text = file_paths[2]
    
    # NOTE: window is from FC to BR, defined using a conservative window from Esamilla, Fleisig (1998)
    release_time_string = f'{ball_release_time - time_offset} {ball_release_time}'
    root.find('.//time_range').text = release_time_string

    # return tree to be written to file
    return tree

# update id setup xml file
def update_id_setup_xml(
    trial_id: str,
    ball_release_time: float,
    throwing_hand: str,
    time_offset: float = 0.217,                         # NOTE: defined based on conservative window from Escamilla, Fleisig (1998)
    xml_path_stem: str = '../../xml_templates'
) -> None:
    # grab & parse correct template file based on throwing hand
    tree = ET.parse(f'{xml_path_stem}/id_{throwing_hand}.xml')
    root = tree.getroot()

    # update file paths
    root.find('.//coordinates_file').text = f'results/{trial_id}_ik.mot'
    root.find('.//output_gen_force_file').text = f'{trial_id}_id.sto'
    
    # update time range 
    release_time_string = f'{ball_release_time - time_offset} {ball_release_time}'
    root.find('.//time_range').text = release_time_string

    # return tree to be written to file
    return tree