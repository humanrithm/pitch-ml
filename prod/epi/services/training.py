import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# random forest model training
def train_rf_model(
        train_data: pd.DataFrame,
        features: list,
        target: str,
        val_data: pd.DataFrame = None,
        params: dict = {'n_estimators': 250, 'random_state': 42},
) -> dict:
    """ 
    Train random forest model and return model, predictions, errors, and RMSE. 
    
    Args:
        train_data (pd.DataFrame): training data
        features (list): list of feature column names
        target (str): target column name
        val_data (pd.DataFrame, optional): validation data. If None, training errors are computed. Defaults to None.
        params (dict, optional): random forest parameters. Defaults to {'n_estimators': 250, 'random_state': 42}.
    Returns:
        dict: {
            'model': trained random forest model,
            'results_on_val_data': 1 if validation data is provided, else 0,
            'predictions': model predictions on validation data or training data,
            'errors': absolute errors on validation data or training data,
            'rmse': root mean squared error on validation data or training data
        }
    """
    
    # fit model to training data
    rf_model = RandomForestRegressor(**params)                            # initialize random forest model
    rf_model.fit(train_data[features], train_data[target])

    # get validation predictions, error (RMSE) in original units
    if val_data is not None:
        predictions = rf_model.predict(val_data[features]) * val_data['height'] * val_data['mass'] * 9.81
        errors = abs(predictions - (val_data[target] * val_data['height'] * val_data['mass'] * 9.81))
        rmse = root_mean_squared_error(
            val_data[target] * val_data['height'] * val_data['mass'] * 9.81,
            predictions
        )

    # else --> compute training errors
    else:
        predictions = rf_model.predict(train_data[features]) * train_data['height'] * train_data['mass'] * 9.81
        errors = abs(predictions - (train_data[target] * train_data['height'] * train_data['mass'] * 9.81))
        rmse = root_mean_squared_error(
            train_data[target] * train_data['height'] * train_data['mass'] * 9.81,
            predictions
        )

    return {
        'model': rf_model,
        'results_on_val_data': 1 if val_data is not None else 0,
        'predictions': predictions,
        'errors': errors,
        'rmse': rmse
    }