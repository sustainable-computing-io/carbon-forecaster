import os
import pandas as pd
import pickle

def forecast_model(model_type, region, steps_into_future):
    filepath = os.path.join(os.path.dirname(__file__), "models/{}_{}.pkl".format(model_type, region))
    with open(filepath, 'rb') as pkl:
        model = pickle.load(pkl)
    fc, confint = model.predict(n_periods=steps_into_future, return_conf_int=True)
    forecast_series = pd.Series(fc, name="forecast")
    lower = pd.Series(confint[:, 0], name="lower_confidence_bound")
    upper = pd.Series(confint[:, 1], name="upper_confidence_bound")
    return pd.concat([forecast_series, lower, upper], axis=1)