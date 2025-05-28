import opensim as osim

def load_osim_model(
        model_path: str
) -> osim.Model:
    model = osim.Model(model_path)
    model.initSystem()

    return model

# update model pose 
def update_model_pose(
        model: osim.Model,
        coords: list = ["arm_add_l", "arm_add_r", "elbow_flex_l", "elbow_flex_r"],
        value: float = 90
    ) -> osim.Model:
    state = model.initSystem()
    
    # create pose angles
    for coord_name in coords:
        coord = model.updCoordinateSet().get(coord_name)
        coord.setLocked(state, False)                           # unlock coordinate
        model.updCoordinateSet().get(coord_name).setValue(state, value)
    
    # update muscle state
    model.equilibrateMuscles(state)

    return model