import traceback
import numpy as np
import pandas as pd
import opensim as osim
from .diff_three_point import diff_three_point
from .filtering import butter_lowpass_filter

class NewtonEuler():
    """
    Inverse Dynamics class for performing top-down or bottom-up inverse dynamics analysis using OpenSim models. Expands upon the OpenSim ID tool by allowing for the extraction of intersegmental moments/forces, such as elbow valgus torque.
    
    Attributes:
        model (osim.Model): The OpenSim model used for the analysis.
        q (pd.DataFrame): Joint angles from inverse kinematics.
        q_dot (pd.DataFrame): Joint angular velocities.
        top_down (bool): Specifies whether to perform top-down or bottom-up inverse dynamics.
        modeled_angles (list[str]): List of joint angles modeled in the analysis.
        motion (dict): Dictionary containing motion data for bodies and joints.
        q (pd.DataFrame): Joint angles.
        q_dot (pd.DataFrame): Joint angular velocities.
        num_samples (int): Number of samples in the data.
        g (np.ndarray): Gravity vector in the world frame.
        forces (dict): Dictionary of forces and moments for each body.
        inertial_params (dict): Dictionary of inertial parameters for each body.
        body_names (list[str]): List of body names in the model.
        body_sequence (list[osim.Body]): Sequence of bodies for the analysis.
        joints (dict): Dictionary of joints in the model.
        child_joints (dict): Dictionary mapping child joints to their names.
        parent_joints (dict): Dictionary mapping parent joints to their names.
        joint_mapping (dict): Dictionary mapping parent joints to child joints.
    
    Methods:
        __init__(model, q, q_dot, top_down): Initializes the InverseDynamics class.
        compute_accel(v, filter_cutoff): Computes acceleration with a filter for noisy data.
        create_forces_dict(num_samples, body_names): Initializes a dictionary of forces and moments.
        create_inertial_params(body_names): Creates inertial parameters for each body.
        create_motion_dict(num_samples, body_names, joint_names): Creates a motion dictionary.
        dcmrot(R, v, mode): Rotates a matrix into a different frame using a rotation matrix.
        flip_body_sequence(body_sequence): Reverses a body sequence for top-down inverse dynamics.
        get_elbow_moments(): Extracts elbow moments from the results.
        get_parent_joint_pos_orient(bname, pjoint, inertial_params): Gets the position and orientation of the parent joint.
        run(): Runs the inverse dynamics analysis.
        skew(vector): Creates a skew symmetric matrix from a vector.
        update_acceleration(motion, filter_cutoff): Updates acceleration values in the motion dictionary.
        update_model_motion(model, state, q, q_dot, coordinates, modeled_angles): Updates the model motion for all time points.
        __setup_bodies_and_joints(top_down): Sets up bodies and joints for the model.
    """

    __version__ = '0.3.0'
    __kinematic_filter__ = 18         # optional freq. for certain acceleration data
    
    # all joint rotations wrt. parent (updated in v0.2.8)
    __rotation_matrices__ = {
        'ground_humerus_l': np.diag([1, 1, 1]),
        'ground_humerus_r': np.diag([1, 1, 1]),
        'elbow_l': np.array([
            [ 9.74103321e-01, -5.03508750e-03,  2.26047270e-01],
            [-5.52972205e-09,  9.99752015e-01,  2.22689894e-02],
            [-2.26103340e-01, -2.16922977e-02,  9.73861758e-01]
        ]),
        'elbow_r': np.array([
            [ 9.74103321e-01, -5.03508750e-03,  2.26047270e-01],
            [-5.52972205e-09,  9.99752015e-01,  2.22689894e-02],
            [-2.26103340e-01, -2.16922977e-02,  9.73861758e-01]
        ]),
        'radioulnar_l': np.array([
            [0.03464939, -0.99939953, 0],
            [0.99939953, 0.03464939, 0],
            [0, 0, 1]
        ]),
        'radioulnar_r': np.array([
            [0.03464939, -0.99939953, 0],
            [0.99939953, 0.03464939, 0],
            [0, 0, 1]
        ]),
        'radius_hand_l': np.array([
            [-3.67320510e-06,  1.00000000e+00,  0.00000000e+00],
            [ 3.67320510e-06,  1.34924357e-11,  1.00000000e+00],
            [ 1.00000000e+00,  3.67320510e-06, -3.67320510e-06]
        ]),
        'radius_hand_r': np.array([
            [-3.67320510e-06,  1.00000000e+00,  0.00000000e+00],
            [ 3.67320510e-06,  1.34924357e-11,  1.00000000e+00],
            [ 1.00000000e+00,  3.67320510e-06, -3.67320510e-06]
        ])
    }

    # specify joint rotation squences by type for euler conversion (v0.2.8)
    __rotation_sequences__ = {
        'customjoint': 'XYZ',
        'pinjoint': 'Z',
        'universaljoint': 'XY',
    }
    
# joint angles associated with each joint (updated in v0.2.8)
    __joint_angles__ = {
        # v0.2.9: sequence is ZXY
        'ground_humerus_l': {
            'Z': 'arm_flex_l',
            'X': 'arm_add_l',
            'Y': 'arm_rot_l'
        },
        'ground_humerus_r': {
            'Z': 'arm_flex_r',
            'X': 'arm_add_r',
            'Y': 'arm_rot_r'
        },
        'elbow_l': {'Z': 'elbow_flex_l'},
        'elbow_r': {'Z': 'elbow_flex_r'},
        'radioulnar_l': {'Z': 'pro_sup_l'},
        'radioulnar_r': {'Z': 'pro_sup_r'},
        'radioulnar_r': {'Z': 'pro_sup_r'},
        'radius_hand_l': {
            'X': 'wrist_flex_l',
            'Y': 'wrist_dev_l'
        },
        'radius_hand_r': {
            'X': 'wrist_flex_r',
            'Y': 'wrist_dev_r'
        }
    }

    # all joint translations wrt. parent (updated in v0.2.8)
        # these are 0 because the jcs are on top of each ohter and joints (non-humerus) do not translate
        # TODO: humerus should be based on tx, ty, tz? but non-consequential for top-down ID calcs
    __translations__ = {
        'ground_humerus_l': np.array([0, 0, 0]),
        'ground_humerus_r': np.array([0, 0, 0]),
        'elbow_l': np.array([0, 0, 0]),
        'elbow_r': np.array([0, 0, 0]),
        'radioulnar_l': np.array([0, 0, 0]),
        'radioulnar_r': np.array([0, 0, 0]),
        'radius_hand_l': np.array([0, 0, 0]),
        'radius_hand_r': np.array([0, 0, 0])
    }
    
    # initialize class w/ osim model, joint angles from IK, and joint angular velocities (to update model)
        # need to pass top_down parameter to specify top-down or bottom-up ID (used to create body set order)
    def __init__(
            self,
            model: osim.Model,
            throwing_hand: str,
            q: pd.DataFrame,            # joint angles
            q_dot: pd.DataFrame,        # joint angular velocities
            top_down: bool,
            sampling_freq: float = (1/480)
    ):
        # initialize model (in case it's not already)
        model.initSystem()

        # set throwing hand
        self.throwing_hand = throwing_hand
        
        # set opensim model params
        self.modeled_angles = q.columns[2:]
        self.model, motion = self.update_model_motion(
            model,
            model.getWorkingState(),        # pass current state
            q, 
            q_dot, 
            model.getCoordinateSet(),
        )

        # set joint coord. data
        self.q = q
        self.q_dot = q_dot
        self.sampling_freq = sampling_freq

        # initialize no. of samples, g
        self.num_samples = len(q)
        self.g = np.array([0, -9.80665, 0])  # world frame
        
        # update motion dictionary w/ acceleration, set in class
        self.motion = self.update_acceleration(motion)

        # set up all bodies & joints
        self.__setup_bodies_and_joints(top_down)

        # initialize forces and inertial parameter dictionaries
        self.forces = self.create_forces_dict(len(q), self.body_names)
        self.inertial_params = self.create_inertial_params(self.body_names)

    # compute acceleration (linear or angular) w/ filter for noisy data
        # compute acceleration values
        # option to apply filter
    def compute_accel(
            self,
            v: np.ndarray,
            filter_cutoff: float = __kinematic_filter__
    ) -> np.ndarray:
        v_df = pd.DataFrame(v.T, columns=['x', 'y', 'z'])                               # butter filter needs columns
        a_df = diff_three_point(v_df)
        
        if filter_cutoff:
            a_df.insert(0, 'time', self.q['time'])
            a_df = butter_lowpass_filter(a_df, cutoff=filter_cutoff)                    # filter noise from accelerations --> avoids inflated moments
        
        return a_df.values.T                                                            # array shape: (3, num_samples)
    
    # convert orientation to Euler angles (XYZ sequence)
        # see 01a_updated_rotation_matrices.ipynb for derivation
    def convert_euler(
            self,
            orientation: np.ndarray,
            sequence: str
    ) -> np.ndarray:
        match sequence.upper():
            case 'XYZ':
                # extract x, y, z from array
                x = orientation[0]
                y = orientation[1]
                z = orientation[2]

                return np.array([
                    [np.cos(y)*np.cos(z), -np.sin(z)*np.cos(y), np.sin(y)],
                    [np.sin(x)*np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x), np.cos(x)*np.cos(z) - np.sin(x)*np.sin(y)*np.sin(z), -np.sin(x)*np.cos(y)],
                    [np.sin(x)*np.sin(z) - np.cos(x)*np.sin(y)*np.cos(z), np.sin(x)*np.cos(z) + np.cos(x)*np.sin(y)*np.sin(z), np.cos(x)*np.cos(y)]
                ])
            
            case 'ZXY':
                return np.array([
                    [-np.sin(x)*np.sin(y)*np.sin(z) + np.cos(x)*np.cos(z), -np.sin(z)*np.cos(x), np.sin(x)*np.sin(z)*np.cos(y) + np.sin(y)*np.cos(z)],
                    [np.sin(x)*np.sin(y)*np.cos(z) + np.sin(z)*np.cos(y), np.cos(x)*np.cos(z), -np.sin(x)*np.cos(y)*np.cos(z) + np.sin(y)*np.sin(z)],
                    [-np.sin(y)*np.cos(x), np.sin(x), np.cos(x)*np.cos(y)]
                ])

            case 'XY':
                # extract x, y from array
                x = orientation[0]
                y = orientation[1]

                return np.array([
                    [np.cos(y), 0, np.sin(y)],
                    [np.sin(x)*np.sin(y), np.cos(x), -np.sin(x)*np.cos(y)],
                    [-np.sin(y)*np.cos(x), np.sin(x), np.cos(x)*np.cos(y)]
                ])
            
            case 'Z':
                # extract z from array
                z = orientation[0]

                return np.array([
                    [np.cos(z), -np.sin(z), 0],
                    [np.sin(z), np.cos(z), 0],
                    [0, 0, 1]
                ])
    
    # initialize dictionary of forces and moments
    def create_forces_dict(
        self,
        num_samples: int, 
        body_names: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        return {
            name: {
                'fnet': np.zeros((3, num_samples)), 
                'mnet': np.zeros((3, num_samples))
                } 
            for name in body_names
        }

    # create inertial parameters
        # add mass of regulation baseball (0.145 kg) to hand, but don't change inertia tensor
    def create_inertial_params(
        self,
        body_names: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        return {
            name: {
                'mass': self.model.getBodySet().get(name).get_mass(), 
                'com': np.array([self.model.getBodySet().get(name).get_mass_center().get(i) for i in range(3)]), 
                'inertia': np.diag([self.model.getBodySet().get(name).get_inertia().get(i) for i in range(3)])
                } 
            for name in body_names
        }

    # create motion dictionary given number of samples
    def create_motion_dict(
            self,
            num_samples: int, 
            body_names: list[str],
            joint_names: list[str]
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        return {
            'body': {
                name: {
                    'orientation': np.zeros((3, 3, num_samples)), 
                    'position': np.zeros((3, num_samples)), 
                    'angular_velocity_in_body': np.zeros((3, num_samples)), 
                    'angular_acceleration_in_body': np.zeros((3, num_samples)),
                    
                    # ground attributes
                    'ground': {
                        'velocity': np.zeros((3, num_samples)), 
                        'angular_velocity': np.zeros((3, num_samples)), 
                        'acceleration': np.zeros((3, num_samples)),
                        'angular_acceleration': np.zeros((3, num_samples)),
                    },
                    
                    # CoM attributes
                    'com': {
                        'position': np.zeros((3, num_samples)), 
                        'velocity': np.zeros((3, num_samples)),
                        'acceleration': np.zeros((3, num_samples))
                        }
                    } 
                for name in body_names},
            'joint': {
                name: {
                    # these are set using results from update_jcs
                    'type': None,               # will be a string
                    'is_child': None,           # will be boolean
                    'child_joints': None,       # will be a list
                    'child': None,              # will be a dictionary
                    'parent': None,             # will be a dictionary
                    
                    # these are set directly in update_motion
                    'rotation': np.zeros((3, 3, num_samples)), 
                    'translation': np.zeros((1, 3, num_samples)),
                    'intersegmental_force': np.zeros((3, num_samples)), 
                    'intersegmental_moment': np.zeros((3, num_samples))
                    } 
                for name in joint_names}
            }
    
    # rotate a matrix into a different frame using a rotation matrix
    def dcmrot(
            self,
            R: np.ndarray,
            v: np.ndarray,
            mode='inv'
    ) -> np.ndarray:
        if mode == 'inv':
            return R.T @ v
        else:
            return R @ v
    
    # reverse a body sequence (eg., for top down ID)
    def flip_body_sequence(
            self,
            body_sequence: list
    ) -> list[osim.Body]:
        body_sequence.reverse()
        return body_sequence

    # extract elbow moments from results
    def get_elbow_moments(self) -> pd.DataFrame:
        # get elbow moments
        match self.throwing_hand:
            case 'right':
                elbow_moments = pd.DataFrame(self.motion['joint']['elbow_r']['intersegmental_moment']).T
            case 'left':
                elbow_moments = pd.DataFrame(self.motion['joint']['elbow_l']['intersegmental_moment']).T
        
        # set columns
        elbow_moments.columns = ['elbow_torque_x', 'elbow_torque_y', 'elbow_torque_z']

        # insert trial info from IK data
        elbow_moments.insert(0, 'time', self.q['time'])
        elbow_moments.insert(0, 'study_id', self.q['study_id'])

        return elbow_moments

    # run (top down) ID by computing Newton Euler cartesian forces for each body and timepoint
    def run(self):

        # iterate through bodies
        for b in range(len(self.body_sequence)):
            # initialize forces, moments
            F_net = np.zeros((3, self.num_samples))
            M_net = np.zeros((3, self.num_samples))
            
            # get body name, motion
            bname = self.body_sequence[b].getName()
            bmotion = self.motion['body'][bname]

            # get inertial terms
            I = self.inertial_params[bname]['inertia']   
            mass = self.inertial_params[bname]['mass']
            p_com = self.inertial_params[bname]['com']

            # get parent joint & set position, orientation of its child
                # (v0.2.6): removed reference to get_parent_pos_orientation function (incorrect)
                # (v0.2.6): subtracted position of center of mass from p_pj in body frame
                # (v0.2.8) confirmed these are constant
            pjoint = self.joints[self.child_joints[bname]]                                      # joint object itself
            p_pj = self.motion['joint'][pjoint.getName()]['child']['position'] - p_com          # position of child joint in body frame relative to CoM
            R_pj = self.motion['joint'][pjoint.getName()]['child']['orientation']               # orientation of child joint in body frame, converted to Euler rotation matrix (XYZ, can hardcode -- constant)
            
            # set linear/angular accel variables
            a_com = bmotion['com']['acceleration']
            w = bmotion['angular_velocity_in_body']
            w_dot = bmotion['angular_acceleration_in_body']
            
            # update force, moment w/ mechanics
            F_inert = mass * a_com                                          # world frame 
            M_inert = (I @ w_dot).T + np.cross(w.T, (I @ w).T)              # body frame
            
            # update net force
                # QA: F_net should not be 0
                # new in v3.0.0 (4/23/25): looping through t preserves shape
            R = bmotion['orientation']
            F_net = np.zeros((3, self.num_samples))
            for t in range(self.num_samples):
                F_net[:, t] = mass * (R[:, :, t] @ self.g)
        
            # check for child joint(s) to accumulate forces
            joint_name = self.child_joints[bname]
            joint_info = self.motion['joint']
            if joint_info[joint_name]['is_child']:
                try:
                    cjoint_names = joint_info[joint_name]['child_joints']   # get all child joint names (should be length 1 no matter what)

                    if len(cjoint_names) > 1:
                        raise ValueError(f"More than one child joint found for {joint_name}.")
                    
                    # iterate through child joints
                        # for example, elbow_r:
                            # cjoint_names to iterate: ['radioulnar_r']
                            # parent body: humerus_r
                            # child body: ulna_r
                    for j in range(len(cjoint_names)):
                        # get joint coordinate system info
                        pjcs = joint_info[cjoint_names[j]]['parent']['position']        # "this body is parent of child joint" 
                        Rjcs = joint_info[cjoint_names[j]]['parent']['orientation']   

                        # joint kinematics
                        Rj = joint_info[cjoint_names[j]]['rotation']              # this varies with time             
                        dj = joint_info[cjoint_names[j]]['translation']           # v0.2.6 --> translation; should be all 0's

                        # get intersegmental force/moment on body in body frame  
                            # negative here is because of Newton's Third Law
                            # new in v0.2.9: moved off dcmrot for handling 3D arrays
                        Fj = np.zeros((3, self.num_samples))
                        for t in range(self.num_samples):
                            Fj[:, t] = -Rj[:, :, t] @ joint_info[cjoint_names[j]]['intersegmental_force'][:, t]    # v3.0.0: removed .T, placed negative

                        Mj = np.zeros((3, self.num_samples))
                        for t in range(self.num_samples):
                            Mj[:, t] = -Rj[:, :, t] @ joint_info[cjoint_names[j]]['intersegmental_moment'][:, t]   # v3.0.0: removed .T, placed negative
                                                
                        # locate Fj point of application
                            # QA: Rjcs @ dj should be 0; removed for shape
                            # (v0.2.6) needed to use com position in body frame (p_com), not from motion; should remove the need for np.tile
                        r = pjcs - p_com

                        # accumulate net force/moment
                            # corrected in v0.2.9 with Einstein sum above
                        F_net += Fj
                        M_net += Mj + np.cross(r, Fj.T).T

                except Exception as e:
                    print(f"Error occurred at {cjoint_names[j]}: {e}")
                    traceback.print_exc()   
            
            # calculate intersegmental force & moment (due to force) at parent joint
                # F_inert and F_net not in the same frame --> need to rotate F_inert into body frame
                # reset Fj for each body
            Fj = np.zeros((3, self.num_samples))
            for t in range(self.num_samples):
                Fj[:, t] = (R[:, :, t].T @ F_inert[:, t]) - F_net[:, t]
            
            # calc intersegmental moment at parent joint
            M_net += self.skew(p_pj) @ Fj
            Mj = M_inert - M_net.T

            # express wrt parent jcs and store 
            self.motion['joint'][pjoint.getName()]['intersegmental_force'] = R_pj.T @ Fj
            self.motion['joint'][pjoint.getName()]['intersegmental_moment'] = R_pj.T @ Mj.T     # this shape is valid; M just got flipped somewhere
    
    # create a skew symmetric matrix from a vector
    def skew(
            self,
            vector: np.ndarray
    ) -> np.ndarray:
        if len(vector) != 3:
            raise ValueError("Input vector must have exactly 3 elements.")

        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
            ])
    
    
    # update acceleration values in a given motion dictionary (velocity values filtered)
    def update_acceleration(
            self,
            motion: dict[str, dict[str, dict[str, np.ndarray]]],
            filter_cutoff: float = None
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        # iterate over bodies
        for body in motion['body'].keys():
            # linear acceleration
            v_ground = motion['body'][body]['ground']['velocity']
            motion['body'][body]['ground']['acceleration'] = self.compute_accel(v_ground, filter_cutoff)
            
            # angular acceleration in ground
            w_ground = motion['body'][body]['ground']['angular_velocity']
            motion['body'][body]['ground']['angular_acceleration'] = self.compute_accel(w_ground, filter_cutoff)

            # angular acceleration in body
            w_body = motion['body'][body]['angular_velocity_in_body']
            motion['body'][body]['angular_acceleration_in_body'] = self.compute_accel(w_body, filter_cutoff)

            # CoM acceleration
            v_com = motion['body'][body]['com']['velocity']
            motion['body'][body]['com']['acceleration'] = self.compute_accel(v_com, filter_cutoff)

        return motion

    # update joint coordinate system (jcs) parameters for a given joint
    def update_jcs(
            self,
            joint: osim.Joint,
            joint_name: str
    ) -> dict: 
        # initialize joint info
        joint_info = {
            'parent': {},
            'child': {}
        }

        # get joint type
        joint_info['type'] = joint.getConcreteClassName()
        
        # set child flag (true if not hand)
        if joint_name not in ['radius_hand_r', 'radius_hand_l']:
            joint_info['is_child'] = True
        else:
            joint_info['is_child'] = False

        # add child joints to each joint
            # NOTE: changed to 1 joint each on 3/26/25
        child_joints = []       # empty list (hand will stay empty)
        match joint_name[:-2]:  # remove _l or _r
            case 'radioular':
                child_joints = ['radius_hand']
            case 'elbow':
                child_joints = ['radioulnar']
            case 'ground_humerus':
                child_joints = ['elbow']
        
        # if present, update child joints with _l or _r based on throwing hand
        if len(child_joints) > 0:
            match self.throwing_hand:
                case 'left':
                    child_joints = [cj + '_l' for cj in child_joints]   # add _l back
                case 'right':
                    child_joints = [cj + '_r' for cj in child_joints]   # add _r back

        joint_info['child_joints'] = child_joints

        # get parent/child names
        parent_offset_name = joint.getPropertyByName('socket_parent_frame').toString()
        child_offset_name = joint.getPropertyByName('socket_child_frame').toString()
        
        # update parent/child bodies
        for j in range(0, 2):
            # determine which segment you're updating (generally, 0:parent, 1:child)
            if parent_offset_name == joint.get_frames(j).getName():
                seg = 'parent'
            elif child_offset_name == joint.get_frames(j).getName():
                seg = 'child'

            # get body, translation of parent/child
            joint_info[seg]['body'] = joint.get_frames(j).findBaseFrame().getName()
            joint_info[seg]['position'] = np.array([joint.get_frames(j).get_translation().get(i) for i in range(3)])
            
            # get orientation (need to convert to Euler)
            orientation = np.array([joint.get_frames(j).get_orientation().get(i) for i in range(3)])
            joint_info[seg]['orientation'] = self.convert_euler(orientation, sequence='XYZ')                    # convert to Euler angles

        return joint_info
    
    # update model motion (coords & bodies) for all time points
    def update_model_motion(
            self,
            model: osim.Model,
            state: osim.State,
            q: pd.DataFrame,
            q_dot: pd.DataFrame,
            coordinates: list[osim.Coordinate],
            modeled_angles: list[str] = [
                'arm_flex', 
                'arm_add',
                'arm_rot',
                'elbow_flex',
                'pro_sup',
                'wrist_flex',
                'wrist_dev'
        ],
    ) -> tuple[
        osim.Model, dict[str, dict[str, dict[str, np.ndarray]]]
    ]:
        # get number of timepoints in trial
        num_samples = len(q)

        # initialize motion dictionary
        body_names = [body.getName() for body in model.getBodySet()]        # get body names
        joint_names = [joint.getName() for joint in model.getJointSet()]    # get joint names
        motion = self.create_motion_dict(num_samples, body_names, joint_names)

        # update modeled angles w/ throwing hand
        # update joint angle cols based on throwing hand
        match self.throwing_hand:
            case 'left':
                modeled_angles = [col + '_l' for col in modeled_angles]
            case 'right':
                modeled_angles = [col + '_r' for col in modeled_angles]
        
        # iterate over samples
        for k in range(num_samples):
            
            # iterate through model coordinates at each time point
            for _, coord in enumerate(coordinates):
                # check if coordinate was saved in IK results
                if coord.getName() in modeled_angles:

                    # get joint angle, velocity (in degrees)
                    q_val = q.loc[k, coord.getName()]
                    q_dot_val = q_dot.loc[k, 'diff_' + coord.getName()]

                    # save joint angle, velocity (in radians)
                    coord.setValue(state, q_val)
                    coord.setSpeedValue(state, q_dot_val)

            # update model state
                # moved outside loop
                # also added acceleration, dynamics for CoM acceleration calcs
            model.realizePosition(state)
            model.realizeVelocity(state)
            model.realizeAcceleration(state)

            # compute kinematics for model bodies at each time point
            for j in range(len(body_names)):
                
                # get body transform in ground
                body_name = body_names[j]
                body = model.getBodySet().get(body_name)
                transform = body.getTransformInGround(state)

                """ get body position attributes and convert to arrays """
                # orientation (orig. mat33) 
                orientation = np.array([[transform.R().get(i, j) for j in range(3)] for i in range(3)])
                motion['body'][body_names[j]]['orientation'][:, :, k] = orientation

                # position (orig. vec3)
                position = np.array([transform.p().get(i) for i in range(3)])
                # position = np.array([body.getPositionInGround(state).get(i) for i in range(3)])
                motion['body'][body_names[j]]['position'][:, k] = position

                # linear velocity (orig. vec3)
                linear_velocity = np.array([body.getLinearVelocityInGround(state).get(i) for i in range(3)])
                motion['body'][body_names[j]]['ground']['velocity'][:, k] = linear_velocity
                
                # angular velocity (orig. vec3)
                angular_velocity_in_ground = np.array([body.getAngularVelocityInGround(state).get(i) for i in range(3)])
                motion['body'][body_names[j]]['ground']['angular_velocity'][:, k] = angular_velocity_in_ground
                
                # angular velocity in body := transpose of orientation mult. w/ angular velocity
                angular_velocity_in_body = orientation.T @ angular_velocity_in_ground
                motion['body'][body_names[j]]['angular_velocity_in_body'][:, k] = angular_velocity_in_body

                # CoM position
                mass_center = np.array([body.get_mass_center().get(i) for i in range(3)])
                com_position = position + (orientation @ mass_center)
                motion['body'][body_names[j]]['com']['position'][:, k] = com_position
                
                # CoM velocity
                    # changed to angular_velocity_in_body (2/26)
                com_velocity = linear_velocity + (orientation @ np.cross(angular_velocity_in_body, mass_center))
                motion['body'][body_names[j]]['com']['velocity'][:, k] = com_velocity

            # update joints
            for j in range(len(joint_names)):
                joint_name = joint_names[j]
                joint = model.getJointSet().get(joint_name)

                # update joint coordinate system info
                jcs_info = self.update_jcs(joint, joint_name)

                # get joint rotation matrix for **motion** (v0.2.8)
                # get rotation sequence
                if joint_name in ['elbow_l', 'elbow_r', 'radioulnar_l', 'radioulnar_r']:
                    rotation_sequence = 'Z'          # elbow is a pin joint
                elif joint_name in ['radius_hand_l', 'radius_hand_r']:
                    rotation_sequence = 'XY'
                elif joint_name in ['ground_humerus_l', 'ground_humerus_r']:
                    rotation_sequence = 'ZXY'
                
                match rotation_sequence:
                    case 'ZYX':
                        # get the angle corresponding to each axis
                        x_angle = q.loc[k, self.__joint_angles__[joint_name]['X']]
                        y_angle = q.loc[k, self.__joint_angles__[joint_name]['Y']]
                        z_angle = q.loc[k, self.__joint_angles__[joint_name]['Z']]
                        
                        # add to dictionary
                        motion['joint'][joint_name]['rotation'][:, :, k] = self.convert_euler(np.array([x_angle, y_angle, z_angle]), sequence='ZYX')
                    
                    case 'XY':
                        # get the angle corresponding to each axis
                        x_angle = q.loc[k, self.__joint_angles__[joint_name]['X']]
                        y_angle = q.loc[k, self.__joint_angles__[joint_name]['Y']]

                        # add to dictionary
                        motion['joint'][joint_name]['rotation'][:, :, k] = self.convert_euler(np.array([x_angle, y_angle]), sequence='XY')
                    
                    case 'Z':
                        # get the angle corresponding to each axis
                        z_angle = q.loc[k, self.__joint_angles__[joint_name]['Z']]

                        # add to dictionary
                        motion['joint'][joint_name]['rotation'][:, :, k] = self.convert_euler(np.array([z_angle]), sequence='Z')
                

                # get joint translation
                    # these are 0 because the jcs are on top of each other (non-translational)
                    # TODO: set at each timepoint (to 0)
                motion['joint'][joint_name]['translation'][:, :, k] = np.array([0, 0, 0])
                motion['joint'][joint_name]['type'] = jcs_info['type']                          # joint type (info. purposes only)
                motion['joint'][joint_name]['is_child'] = jcs_info['is_child']                  # boolean
                motion['joint'][joint_name]['child_joints'] = jcs_info['child_joints']          # list of child joints
                motion['joint'][joint_name]['child'] = jcs_info['child']                        # this sets all child info: body, translation, and Euler angle rotation
                motion['joint'][joint_name]['parent'] = jcs_info['parent']                      # this sets all parent info: body, translation, and Euler angle rotation
                
        return model, motion

    # set up bodies and joints for model
    def __setup_bodies_and_joints(
            self,
            top_down: bool
    ) -> None:
        # set up body sequence
        body_sequence = list(self.model.getBodySet())
        self.body_names = [body.getName() for body in self.model.getBodySet()]

        # flip body sequence if top down ID
        if top_down:
            self.body_sequence = self.flip_body_sequence(body_sequence)

        # get child joints for all bodies
        self.joints = {j.getName(): j for j in self.model.getJointSet()}
        self.child_joints = {j.getChildFrame().getName().rstrip('_offset'): j.getName() for j in self.model.getJointSet()}
        self.parent_joints = {j.getParentFrame().getName().rstrip('_offset'): j.getName() for j in self.model.getJointSet()}
        self.joint_mapping = {parent: child for parent, child in zip(self.parent_joints.keys(), self.child_joints.keys())}                 # parent: child joints

        