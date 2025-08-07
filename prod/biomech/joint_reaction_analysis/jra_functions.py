import copy
import opensim
import numpy as np
import pandas as pd
from biomech.algorithms import diff_three_point

# force set handling
def handle_force_set(
        model: opensim.Model, 
        remove_spheres: bool = True
) -> tuple[opensim.Model, opensim.ForceSet]:
    """
    Handle the force set of the model by removing spheres if specified.
    
    Args:
        model (opensim.Model): The OpenSim model to modify.
        remove_spheres (bool): Whether to remove spheres from the force set.
    
    Returns:
        opensim.Model: The modified OpenSim model.
    """
    force_set = model.getForceSet()
    i = 0

    # iterate through all of the force set
    while i < force_set.getSize():
        if remove_spheres:
            force_set.remove(i)
        else:
            if 'SmoothSphere' not in force_set.get(i).getConcreteClassName():
                force_set.remove(i)
            else:
                i += 1
    
    return model, force_set

# handle all coordinates in the model
def handle_coordinates(
        model: opensim.Model
) -> tuple[opensim.CoordinateSet, int, list, opensim.ControllerSet]:
    """ 
    Handle the coordinates of the model by adding coordinate actuators and controllers.
    
    Args:
        model (opensim.Model): The OpenSim model to modify. 
    Returns:
        tuple: A tuple containing:
            - coords (opensim.CoordinateSet): The set of coordinates in the model.
            - n_coords (int): The number of coordinates.
            - coord_names (list): A list of coordinate names.
            - controller_set (opensim.ControllerSet): The set of controllers in the model.
    
    """
    
    # get all coordinates in model
    coords = model.getCoordinateSet()
    n_coords = coords.getSize()
    coord_names = [coords.get(i).getName() for i in range(n_coords)]

    # add coordinate actuators
    actuatorNames = []
    for coord in coords:
        newActuator = opensim.CoordinateActuator(coord.getName())
        newActuator.setName(coord.getName() + '_actuator')
        actuatorNames.append(coord.getName() + '_actuator')
        newActuator.set_min_control(-np.inf)
        newActuator.set_max_control(np.inf)
        newActuator.set_optimal_force(1)
        model.addForce(newActuator)
        
        # add prescribed controllers for coordinate actuators / construct constant function.
            # NOTE: neded for joint reaction analysis to work properly        
        constFxn = opensim.Constant(0) 
        constFxn.setName(coord.getName() + '_constFxn')         
        
        # construct prescribed controller.
        pController = opensim.PrescribedController() 
        pController.setName(coord.getName() + '_controller') 
        pController.addActuator(newActuator)
        
        # attach the function to the controller.
        pController.prescribeControlForActuator(0,constFxn) 
        model.addController(pController) 

    # get controller set
    controller_set = model.getControllerSet()

    return coords, n_coords, coord_names, controller_set

# update kinematic states with coordinates and IK results file
def update_kinematic_states(
        n_coords: int,
        coords: opensim.CoordinateSet,
        coord_names: list,
        ik_path: str = 'trial_motion.mot',
        preset_degree_state: bool = True,
        Qds: np.ndarray = np.array([]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, list]: 
    """ 
    Update kinematic states from the input motion file. Includes `q` and `qdot` calculations.

    Args:
        n_coords (int): The number of coordinates in the model.
        coords (opensim.CoordinateSet): The set of coordinates in the model.
        coord_names (list): A list of coordinate names.
        ik_path (str): Path to the input motion file containing kinematic states.
        preset_degree_state (bool): Whether to use a preset value for the degree state.  
        Qds (np.ndarray): The time derivatives of the kinematic states (joint velocities). If empty, they will be computed. 
    Returns:
        tuple: A tuple containing:
            - q (np.ndarray): The kinematic states (joint angles).
            - qd (np.ndarray): The time derivatives of the kinematic states (joint velocities).
            - state_time (np.ndarray): The time vector for the kinematic states.
            - in_degrees (bool): Whether the kinematic states are in degrees.
            - state_names (list): The names of the kinematic states.
    """
    # load kinematic states
    state_table = opensim.TimeSeriesTable(ik_path)
    state_names = state_table.getColumnLabels()
    state_time = state_table.getIndependentColumn()

    # execute degrees check
    try:
        in_degrees = state_table.getTableMetaDataAsString('inDegrees') == 'yes'
    except:
        in_degrees = preset_degree_state
        print('Using preset degree state variable: {}'.format(preset_degree_state))

    # update q, qdot
        # NOTE: tbd if Qds should be 0
    q = np.zeros((len(state_time), n_coords))
    dt = state_time[1] - state_time[0]    
    if len(Qds) > 0:
        qd_t = np.zeros((len(state_time), n_coords))

    for col in state_names:
            # remove activiation columns 
            if 'activation' in col:
                state_table.removeColumn(col)
            
            else:
                coordCol = coord_names.index(col)
                
                # convert to radians if in degrees
                for t in range(len(state_time)):
                    qTemp = np.asarray(state_table.getDependentColumn(col)[t])                
                    if coords.get(col).getMotionType() == 1 and in_degrees:
                        qTemp = np.deg2rad(qTemp) # convert rotation to rad.
                    q[t,coordCol] = copy.deepcopy(qTemp)
                
                # if qd_t is not None, update qd_t
                if len(Qds) > 0:
                    idx_col = state_names.index(col)
                    qd_t[:,coordCol] = Qds[:, idx_col]

    # compute qdot w/ 3PCD method
    if not len(Qds) > 0:
        qd = diff_three_point(q, dt)
    else:
        qd = qd_t  

    return q, qd, state_time, in_degrees, state_names

# load ID results & get time
def load_id_results(id_path: str = 'trial_moments.sto') -> tuple[opensim.TimeSeriesTable, tuple]:
    id_table = opensim.TimeSeriesTable(id_path)
    id_time = id_table.getIndependentColumn()

    return id_table, id_time

# create JRA setup dictionary
def setup_joint_reaction_analysis(
        model: opensim.Model,
        coord_names: list,
        mot_data: pd.DataFrame,
        jra_path: str = 'setup_jra.xml',
        print_to_xml: bool = True
    ) -> dict:
    """
    Setup the Joint Reaction Analysis (JRA) for the given OpenSim model.
    
    Args:
        model (opensim.Model): The OpenSim model to setup for JRA.
        mot_data (pd.DataFrame): The motion data containing kinematic states.
        jra_path (str): Path to the JRA setup XML file.
        print_to_xml (bool): Whether to print the JRA setup to XML.
    
    Returns:
        dict: A dictionary containing the model, state, system position indices, system velocity indices, 
              state name list, and the JointReaction analysis object.
    """
    # initialize model system (NOTE: done editing model at this point)
    state = model.initSystem()

    # create state Y map 
    y_names = opensim.createStateVariableNamesInSystemOrder(model)
    system_position_idxs = []
    system_velocity_idxs = []
    stateNameList = []
    for stateName in coord_names:
        posIdx = np.squeeze(
            np.argwhere([stateName + '/value' in y for y in y_names]))
        velIdx = np.squeeze(
            np.argwhere([stateName + '/speed' in y for y in y_names])) 
        if posIdx.size>0:  
            system_position_idxs.append(posIdx)
            system_velocity_idxs.append(velIdx)
            stateNameList.append(stateName)

    # setup tool with setup path
    jointReaction = opensim.JointReaction(jra_path)
    model.addAnalysis(jointReaction)
    jointReaction.setModel(model)

    # update start/end times
    jointReaction.setStartTime(mot_data['time'].values[0])
    jointReaction.setEndTime(mot_data['time'].values[-1])

    # reprint to XML if specified
    if print_to_xml:
        jointReaction.printToXML(jra_path)

    return {
        'model': model,
        'state': state,
        'system_position_idxs': system_position_idxs,
        'system_velocity_idxs': system_velocity_idxs,
        'stateNameList': stateNameList,
        'jointReaction': jointReaction
    }

# run JRA by stepping through time
    # NOTE: module will make this much cleaner
def run_joint_reaction_analysis(
        model: opensim.Model,
        state: opensim.State,
        jointReaction: opensim.JointReaction,
        coords: list,
        n_coords: int,
        controller_set: opensim.ControllerSet,
        id_table: opensim.TimeSeriesTable,
        id_time: list,
        state_time: list,
        q: pd.DataFrame,
        qd: pd.DataFrame,
        system_position_idxs: list,
        system_velocity_idxs: list
) -> opensim.JointReaction:
    controls = opensim.Vector(n_coords,0)
    for iTime in range(len(state_time)):
        thisTime = state_time[iTime]    
        if thisTime <= id_time[-1]:             
            
            # set time
            id_row = id_table.getNearestRowIndexForTime(thisTime)  
            state.setTime(thisTime)                
            
            # set state, velocity, actuator controls
            yVec = np.zeros((state.getNY())).tolist()
            
            # loop through states to set values and speeds
            for iCoord, coord in enumerate(coords):
                if '_beta' not in coord.getName():
                    # update yVec with position and velocity
                    yVec[system_position_idxs[iCoord]] = q[iTime,iCoord]
                    yVec[system_velocity_idxs[iCoord]] = qd[iTime,iCoord]                    
                    
                    # set suffix based on motion type
                    if coord.getMotionType() == 1: # rotation
                        suffix = '_moment'
                    elif coord.getMotionType() == 2: # translation
                        suffix = '_force'                        
                    
                    # aet prescribed controller constant value to control value
                        # NOTE: controls don't live through joint reaction analysis.
                    thisController = opensim.PrescribedController.safeDownCast(controller_set.get(coord.getName() + '_controller')) 
                    thisConstFxn = opensim.Constant.safeDownCast(thisController.get_ControlFunctions(0).get(0))
                    thisConstFxn.setValue(id_table.getDependentColumn(coord.getName()+suffix)[id_row])
            
                    # setting controls this way is redundant
                        # however, it's necessary if want to do a force reporter in the future
                    controls.set(iCoord, id_table.getDependentColumn(coord.getName()+suffix)[id_row])

            # set yVec to state and realize velocity
            state.setY(opensim.Vector(yVec))
            model.realizeVelocity(state)                
            model.setControls(state, controls)

            # realize acceleration (to be safe)
            model.realizeAcceleration(state)

        # step through JRA 
            # NOTE: this is outside of everything except time loop
        if iTime == 0:
            jointReaction.begin(state) 
        else:
            jointReaction.step(state, iTime) 
        if iTime == len(state_time)-1 or thisTime >= id_time[-1]:
            jointReaction.end(state)

    return jointReaction

def get_evt_col(throwing_hand: str) -> str:
    """ Get the JRA column name for elbow varus torque based on the throwing hand. """
    if throwing_hand == 'right':
        return 'elbow_r_on_ulna_r_in_ulna_r_mx'
    elif throwing_hand == 'left':
        return 'elbow_l_on_ulna_l_in_ulna_l_mx'
    else:
        raise ValueError(f"Unknown throwing hand: {throwing_hand}")

def postprocess_evt_results(
        data: pd.DataFrame,
        ball_release_time: float,
        foot_contact_time_offset: float = 0.217,
        time_col: str = 'time'
):
    """ 
    Postprocesses the event results for JRA trials. Includes time trimming (FC to BR) and normalization.

    Args:
        data (pd.DataFrame): The event results DataFrame.
        br_time (float): The ball release time.
        fc_offset (float): The offset from ball release to foot contact.

    Returns:
        pd.DataFrame: The postprocessed EVT results DataFrame.
    """

    # calculate foot contact time, then filter to window
    foot_contact_time = ball_release_time - foot_contact_time_offset
    filtered_data = data[(data[time_col] >= foot_contact_time) & (data[time_col] <= ball_release_time)]
    
    # normalize time such that 0 = FC and 1 = BR
    filtered_data.insert(
        1, 
        'normalized_time',
        compute_normalized_time(filtered_data[time_col])
    )

    return filtered_data.reset_index(drop=True)

# create normalized time column (for a single trial)
def compute_normalized_time(
    time: pd.Series
) -> pd.Series:
    """ Compute normalized time (i.e., 0-1) for a given time series. """
    return (time - time.min()) / (time.max() - time.min())

# get peak value summary for a trial
def get_trial_peaks(
        data: pd.DataFrame,
        peak_col: str = 'elbow_varus_torque'
) -> dict:
    data_max = data[peak_col].max()
    data_min = data[peak_col].min()

    if data_max > abs(data_min):
        return {
            'study_id': data['study_id'].unique()[0],
            'peak_value': data[peak_col].max(),
            'peak_time': data['time'][data[peak_col].idxmax()],             
            'peak_normalized_time': data['normalized_time'][data[peak_col].idxmax()],
            'peak_idx': data[peak_col].idxmax(),
            'peak_was_negative': 0
        }
    else:
        return {
            'study_id': data['study_id'].unique()[0],
            'peak_value': abs(data[peak_col].min()),
            'peak_time': data['time'][data[peak_col].idxmin()],             
            'peak_normalized_time': data['normalized_time'][data[peak_col].idxmin()],
            'peak_idx': data[peak_col].idxmin(),
            'peak_was_negative': 1
        }
    
# inspect subject results for outliers
def inspect_subject_results(
        data: pd.DataFrame,
        peak_label: str = 'peak_value'
) -> pd.DataFrame:
    # get peak value mean and standard deviation
        # NOTE: using median for mean to be more robust to outliers
    subject_avg = data[peak_label].median()
    subject_std = data[peak_label].std()

    # add `outlier_flag` column
    data['outlier_flag'] = 0

    # iterate through rows to check for outliers
    for idx, values in data.iterrows():
        # update outlier flag if peak value is more than 2 standard deviations from the mean
            # NOTE: using median for mean to be more robust to outliers
        if (values['peak_value'] > subject_avg + 1.96 * subject_std) or (values['peak_value'] < subject_avg - 1.96 * subject_std):
            data.at[idx, 'outlier_flag'] = 1

    return data


