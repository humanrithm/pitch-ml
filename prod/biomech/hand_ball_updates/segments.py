import numpy as np
import xml.etree.ElementTree as ET

# update scaling tool XML for subject
    # subject_id: subject to scale (info loaded from json)
def update_model_hand_segment(
    tree: ET.ElementTree,
    throwing_hand: str,
    ball_mass: float = 0.145,                                       # mass of ball
    ball_radius: float = 0.0368,                                    # radius of ball
) -> None:

    root = tree.getroot()

    # find the hand body
    match throwing_hand:
        case 'left':
            hand_body = root.find(".//Body[@name='hand_l']")
        case 'right':
            hand_body = root.find(".//Body[@name='hand_r']")
    
    # update mass (+= 0.145)
        # combined center of mass computed with mass weighted avg of two bodies
    hand_mass = hand_body.find(".//mass")
    hand_mass.text = str(float(hand_mass.text) + ball_mass)
    
    # update mass center using mass wtd avgs
    com_new = compute_combined_com(float(hand_mass.text), ball_mass)
    hand_com = hand_body.find(".//mass_center")
    hand_com.text = ' '.join(map(str, com_new)) 

    # update inertia tensor as hand-ball segment
        # uses updated mass, mass center
        # tensor computed using parallel axis theorem
    I_new = compute_combined_inertia(float(hand_mass.text), com_new, m_ball=ball_mass, r_ball=ball_radius)
    hand_inertia = hand_body.find(".//inertia")
    hand_inertia.text = ' '.join(map(str, [I_new[0, 0], I_new[1, 1], I_new[2, 2], 0, 0, 0]))

    return tree

# compute combined CoM using mass wtd avgs
def compute_combined_com(
        m_hand: float,
        m_ball: float = 0.145           # mass of regulation ball in kg
) -> np.ndarray:

    # center of mass
    com_hand = np.array([0, -0.068095, 0])  
    com_ball = np.array([0.0368, -0.068095, 0])                                     # ball CoM shifted in x-direction
    com_new = (m_hand * com_hand + m_ball * com_ball) / (m_hand + m_ball)           # combined CoM

    return com_new

# compute combined inertia tensor using parallel axis theorem
def compute_combined_inertia(
        m_hand: float,
        com_combined: np.ndarray,
        com_hand: np.ndarray = np.array([0, -0.068095, 0]),
        com_ball: np.ndarray = np.array([0.0368, -0.068095, 0]),                    # ball CoM in hand frame shifted in x-direction by radius of ball
        m_ball: float = 0.145,                                                      # mass of regulation ball in kg
        r_ball: float = 0.0368,                                                     # radius of regulation ball in m
):
    # compute updated axes
    com_hand_prime = com_hand - com_combined
    com_ball_prime = com_ball - com_combined

    # inertia tensors (original; hand from rajagopal model)
    I_hand = np.diag([0.00093985399999999995, 
                    0.00057634600000000004, 
                    0.0014118900000000001])
    I_ball = 2/5 * m_ball * r_ball**2 * np.eye(3)  # ball inertia tensor; uniform density sphere

    # update inertia tensors through new axes -- parallel axis theorem
    I_hand_prime = I_hand + m_hand * (skew(com_hand_prime) @ skew(com_hand_prime))
    I_ball_prime = I_ball + m_ball * (skew(com_ball_prime) @ skew(com_ball_prime))

    # combined inertia tensor
    I_new = I_hand_prime + I_ball_prime

    return I_new

# create a skew symmetric matrix from a vector
def skew(vector: np.ndarray) -> np.ndarray:
    if len(vector) != 3:
        raise ValueError("Input vector must have exactly 3 elements.")

    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
        ])
