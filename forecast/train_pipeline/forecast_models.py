from __future__ import annotations
import pandas as pd
import xgboost as xgb
from neuralprophet import NeuralProphet
from abc import ABC, abstractmethod
import os
from pmdarima import auto_arima
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any, Dict
import matplotlib.pyplot as plt
from enum import Enum
import pickle

def close_event():
    plt.close()

class Step_Type(Enum):
    Hour = 1
    Day = 2
    Half_Hour = 3
    Month = 4
    Year = 5
    Auto = 6

class GenericCarbonIntensityForecastModel(ABC):

    # time series dataframe requires datetime and carbon intensity and region
    # precondition: carbon intensity time series data need to belong to a single region
    """Generic Time Series Carbon Forecast Model
    
    Attributes:
        _co2_time_series_dataframe: carbon intensity time series data
        region: region we are forecasting
        step_type: step type of forecaster
        model_type: class name of model
        save_location: file directory to save the model
        filename: name of file model to save
        model: model forecaster
        recent_results: model error and accuracy (empty if none are provided)
    """
    region: str
    step_type: Step_Type
    model_type: str
    __save_location: str
    filename: str
    model: Any
    _co2_time_series_dataframe: pd.DataFrame
    recent_results: Dict[str, Any]

    def __init__(self, region: str, step_type: Step_Type, model_type: str, save_location="default"):
        self.region = region
        self.save_location = save_location
        self.step_type = step_type
        self.filename = "{}_{}.pkl".format(model_type, self.region)
        self.model = None
        self._co2_time_series_dataframe = None
        self.recent_results = {}


    def _validate_and_clean_co2_time_series_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return df
        if "datetime" in df and "carbon_intensity" in df:
            clean_df = pd.DataFrame(columns=['datetime', 'carbon_intensity'])
            clean_df['datetime'] = df['datetime']
            clean_df['carbon_intensity'] = df['carbon_intensity']
                
            return clean_df
        raise Exception("missing required columns: datetime, carbon_intensity")


    @property
    def save_location(self) -> str:
        return self.__save_location


    @save_location.setter
    def save_location(self, folderpath: str):
        if folderpath == "default":
            self.__save_location = os.path.join(os.path.dirname(__file__), "models")
        else:
            if os.path.exists(folderpath):
                self.__save_location = folderpath
            

    @abstractmethod
    def create_new_model(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def _preprocess_co2_time_series_df(self) -> None:
        raise NotImplementedError

    
    @abstractmethod
    def train(self, no_existing_model: bool, new_co2_time_series_dataframe: pd.DataFrame, include_test_df) -> None:
        raise NotImplementedError


    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError 

    
    def delete_model(self) -> None:
        self.model = None

    def forecast(self, steps_into_future: int) -> pd.DataFrame:
        raise NotImplementedError


class NeuralProphetModel(GenericCarbonIntensityForecastModel):
    """Forecaster which uses NeuralProphet

    Attributes:
        region: region we are forecasting
        step_type: step type of forecaster
        model_type: class name of model
        save_location: file directory to save the model
        filename: name of file model to save
        model: model forecaster
        train_df: training dataframe
    """
    train_df: pd.DataFrame    

    step_type_to_neural_prophet_freq = {
        Step_Type.Day: "D",
        Step_Type.Half_Hour: "30min",
        Step_Type.Month: "M",
        Step_Type.Year: "Y",
        Step_Type.Hour: "60min",
        Step_Type.Auto: "auto"
    }
    
    @staticmethod
    def neural_prophet_plot(type_of_loss, training_loss, validation_loss):
        figure = plt.figure(figsize=(12,5), dpi=100)
        timer = figure.canvas.new_timer(interval = 200000)
        timer.add_callback(close_event)
        plt.plot(training_loss, label='training loss')
        plt.plot(validation_loss, label='validation loss', color="green")
        plt.title(type_of_loss)
        plt.legend(loc='upper left', fontsize=8)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        timer.start()
        plt.show()

    def __init__(self, region: str, step_type=Step_Type.Auto, save_location="default"):
        super().__init__(region, step_type, self.__class__.__name__, save_location)
        self.train_df = None


    def __save_co2_time_series_dataframe(self):
        try: 
            self._co2_time_series_dataframe.to_pickle(os.path.join(self.save_location, "{}_{}_df_pkl".format(self.model_type, self.region)))
        except Exception:
            raise Exception("could not save dataframe")


    def __load_co2_time_series_dataframe(self):
        try:
            self._co2_time_series_dataframe = pd.read_pickle(os.path.join(self.save_location, "{}_{}_df_pkl".format(self.model_type, self.region)))
        except Exception:
            raise Exception("could not load dataframe")


    def create_new_model(self) -> None:
        #self.model = NeuralProphet(n_lags=3*24, n_forecasts=3*24, changepoints_range=0.90, n_changepoints=30, batch_size=50)
        self.model = NeuralProphet(n_lags=3*24, n_forecasts=3*24)

    # note that neural prophet must always train from scratch! So appending new data is crucial
    def __append_new_co2_time_series_dataframe(self, df: pd.DataFrame) -> None:
        clean_df = self._validate_and_clean_co2_time_series_df(df)
        self._co2_time_series_dataframe = pd.concat([self._co2_time_series_dataframe, clean_df])
    

    def _preprocess_co2_time_series_df(self) -> None:
        if self._co2_time_series_dataframe.empty or self._co2_time_series_dataframe is None:
            raise Exception("dataframe is empty")
        clean_df = pd.DataFrame(columns=['ds', 'y'])
        clean_df['ds'] = pd.to_datetime(self._co2_time_series_dataframe['datetime']).dt.tz_localize(None)
        clean_df['y'] = self._co2_time_series_dataframe['carbon_intensity']
        clean_df['y'] = clean_df['y'].astype(float)
        clean_df.dropna(inplace=True)
        self.train_df = clean_df
        

    def train(self, no_existing_model: bool, new_co2_time_series_dataframe: pd.DataFrame, include_test_df) -> None:
        if no_existing_model:
            self.create_new_model()
            self._co2_time_series_dataframe = self._validate_and_clean_co2_time_series_df(new_co2_time_series_dataframe)
        else:
            self.__append_new_co2_time_series_dataframe(new_co2_time_series_dataframe)
        self._preprocess_co2_time_series_df()
        print(len(self.train_df))
        train_df, validate_df = self.model.split_df(self.train_df, freq=self.step_type_to_neural_prophet_freq[self.step_type], valid_p=0.2)
        metrics = self.model.fit(train_df, validation_df=validate_df, freq=self.step_type_to_neural_prophet_freq[self.step_type])
        self.recent_results = {"training_loss_MAE": metrics["MAE"], "validation_loss_MAE": metrics["MAE_val"]}
        self.neural_prophet_plot("MAE Loss", metrics["MAE"], metrics["MAE_val"])


    def save(self):
        #self.__save_co2_time_series_dataframe()
        with open(os.path.join(self.save_location, self.filename), "wb") as f:
            pickle.dump(self.model, f)


    def load(self):
        #self.__load_co2_time_series_dataframe()
        self._co2_time_series_dataframe = pd.read_csv(os.path.join(self.save_location, "NeuralProphet.csv"))
        self._preprocess_co2_time_series_df()
        with open(os.path.join(self.save_location, self.filename), "rb") as input_file:
            self.model = pickle.load(input_file)


    def forecast(self, steps_into_future: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(self.train_df.tail(200), n_historic_predictions=steps_into_future)
        forecast = self.model.predict(future)
        return forecast.tail(steps_into_future)


class ARIMAModel(GenericCarbonIntensityForecastModel):
    """Forecaster which uses Traditional ARIMA Model

    Attributes:
        co2_time_series_dataframe: carbon intensity time series data
        region: region we are forecasting
        step_type: step type of forecaster
        model_type: class name of model
        save_location: file directory to save the model
        filename: name of file model to save
        model: model forecaster
        train_df: training dataframe
        test_df: testing dataframe
        recent_results: error and accuracy results of model testing 
    """
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    step_to_m = {Step_Type.Auto: 1, Step_Type.Day: 7, Step_Type.Month: 12, Step_Type.Year: 1, Step_Type.Hour: 24, Step_Type.Half_Hour: 48}

    #TODO: Turn to static method
    def arima_plot_results(self, f, ci): 
        ci_lower = []
        ci_upper = []
        for c in ci:
            ci_lower.append(c[0])
            ci_upper.append(c[1])
        figure = plt.figure(figsize=(12,5), dpi=100)
        timer = figure.canvas.new_timer(interval = 15000)
        timer.add_callback(close_event)
        plt.plot(self.train_df['value'], label='training Co2')
        plt.plot(self.test_df['value'], label='actual Co2', color="green")
        plt.plot(pd.Series(f, index=self.test_df['value'].index), label='forecasted Co2', color="orange")
        plt.fill_between(self.test_df['value'].index, ci_lower, ci_upper, 
                    color='lightgrey', alpha=.15)
        plt.title('Forecasted and Real Carbon Intensity')
        plt.legend(loc='upper left', fontsize=8)
        timer.start()
        plt.show()


    def __init__(self, region: str, step_type=Step_Type.Half_Hour, save_location="default"):
        super().__init__(region, step_type, self.__class__.__name__, save_location)
        self.train_df = None
        self.test_df = None
    

    def create_new_model(self) -> None:
        self._preprocess_co2_time_series_df()
        

    def _preprocess_co2_time_series_df(self) -> None:
        if self._co2_time_series_dataframe.empty or self._co2_time_series_dataframe is None:
            raise Exception("dataframe is empty")
        clean_df = pd.DataFrame(columns=['value'])
        clean_df['value'] = self._co2_time_series_dataframe['carbon_intensity']
        clean_df['value'] = clean_df['value'].astype(float)
        clean_df.set_index(pd.to_datetime(self._co2_time_series_dataframe['datetime']), inplace=True)
        clean_df.dropna(inplace=True)
        split_idx = round(len(clean_df)* 0.8)
        self.train_df = clean_df[:split_idx]
        self.test_df = clean_df[split_idx:]

    
    def __forecast_single_step(self):
        fc, conf = self.model.predict(n_periods=1, return_conf_int=True)
        return (fc.tolist()[0], np.asarray(conf).tolist()[0])

    
    def __forecast_test_dataset(self):
        forecasts = []
        confidence_intervals = []
        for _, t in self.test_df.iterrows():
            fc, conf = self.__forecast_single_step()
            forecasts.append(fc)
            confidence_intervals.append(conf)
            self.model.update(t['value'])
        
        return forecasts, confidence_intervals, mean_absolute_error(pd.Series(self.test_df['value']), forecasts), mean_squared_error(pd.Series(self.test_df['value']), forecasts), r2_score(pd.Series(self.test_df['value']), forecasts)

    
    def train(self, no_existing_model: bool, new_co2_time_series_dataframe: pd.DataFrame, include_test_df) -> None:
        self._co2_time_series_dataframe = self._validate_and_clean_co2_time_series_df(new_co2_time_series_dataframe)
        self.create_new_model()
        if no_existing_model:
            print("Training Now")
            self.model = auto_arima(self.train_df, seasonal=True, m=self.step_to_m[self.step_type])
            f, ci, mae, mse, r2 = self.__forecast_test_dataset()
        else:    
            self.model.update(self.train_df)
            f, ci, mae, mse, r2 = self.__forecast_test_dataset()
        
        self.arima_plot_results(f, ci)

        self.recent_results = {"forecasts": f, "confidence": ci, "mae": mae, "mse": mse, "r2": r2}


    def save(self) -> None:
        # save model
        try:
            with open(os.path.join(self.save_location, self.filename), 'wb') as pkl:
                pickle.dump(self.model, pkl)
        except:
            raise Exception("failed to save model")


    def load(self) -> None:
        # reload model
        try:
            with open(os.path.join(self.save_location, self.filename), 'rb') as pkl:
                self.model = pickle.load(pkl)
        except:
            raise Exception("failed to reload model")
    

    def forecast(self, steps_into_future: int) -> pd.DataFrame:
        fc, confint = self.model.predict(n_periods=steps_into_future, return_conf_int=True)
        forecast_series = pd.Series(fc, name="forecast")
        lower = pd.Series(confint[:, 0], name="lower_confidence_bound")
        upper = pd.Series(confint[:, 1], name="upper_confidence_bound")
        return pd.concat([forecast_series, lower, upper], axis=1)

#TODO: Create abstract class for machine learning models
class XGBoostRegressorModel():
    """Forecaster which uses Extreme Gradient Boosting Machine Learning Regressor

    Attributes:
        co2_time_series_dataframe: carbon intensity time series data
        region: region we are forecasting
        step_type: step type of forecaster
        model_type: class name of model
        save_location: file directory to save the model
        filename: name of file model to save
        model: model forecaster

        train_df_features: training features
        train_df_label: training label (co2 intensity)
        test_df_features: testing features
        test_df_label: testing labels
        recent_results: error and accuracy of model
        
    """
    region: str
    step_type: Step_Type
    model_type: str
    __save_location: str
    filename: str
    model: Any
    _co2_time_series_dataframe: pd.DataFrame
    train_df_features: pd.DataFrame
    train_df_label: pd.DataFrame
    test_df_features: pd.DataFrame
    test_df_label: pd.DataFrame
    recent_results: Dict[str, Any]

    def __init__(self, region: str, step_type: Step_Type, model_type: str, save_location="default"):
        self.region = region
        self.save_location = save_location
        self.step_type = step_type
        self.model = None
        self._co2_time_series_dataframe = None

        self.train_df_features = None
        self.train_df_label = None
        self.test_df_features = None
        self.test_df_label = None
        self.recent_results = None
        self.filename = "{}_{}.txt".format(self.__class__.__name__, self.region)

    def _validate_and_clean_co2_time_series_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return df
        if "datetime" in df and "carbon_intensity" in df:
            clean_df = pd.DataFrame(columns=['datetime', 'carbon_intensity'])
            clean_df['datetime'] = df['datetime']
            clean_df['carbon_intensity'] = df['carbon_intensity']
                
            return clean_df
        raise Exception("missing required columns: datetime, carbon_intensity")


    @property
    def save_location(self) -> str:
        return self.__save_location


    @save_location.setter
    def save_location(self, folderpath: str):
        if folderpath == "default":
            dirname=os.path.dirname
            self.__save_location = os.path.join(os.path.dirname(dirname(__file__)), "models")
        else:
            if os.path.exists(folderpath):
                self.__save_location = folderpath

    def create_new_model(self) -> None:
        self.model = xgb.XGBRegressor(n_estimators=1000)

    @staticmethod
    def generate_features_df(df: pd.DataFrame) -> pd.DataFrame:
        features_df = pd.DataFrame()
        features_df['minute'] = df.index.minute
        features_df['hour'] = df.index.hour
        features_df['dayofweek'] = df.index.dayofweek
        features_df['dayofmonth'] = df.index.dayofmonth
        features_df['dayofyear'] = df.index.dayofyear
        features_df['month'] = df.index.month
        features_df['weekofyear'] = df.index.weekofyear
        features_df['year'] = df.index.year
        return features_df


    def _preprocess_co2_time_series_df(self) -> None:
        if self._co2_time_series_dataframe.empty or self._co2_time_series_dataframe is None:
            raise Exception("dataframe is empty")
        clean_df = pd.DataFrame(columns=['Datetime', 'carbon_intensity'])
        clean_df['Datetime'] = self._co2_time_series_dataframe['datetime']
        clean_df.dropna()
        clean_df.set_index(pd.Series(clean_df['Datetime']), inplace=True)
        clean_df.index = pd.to_datetime(clean_df.index)
        clean_df['carbon_intensity'] = self._co2_time_series_dataframe['carbon_intensity']
        split_idx = round(len(clean_df)* 0.8)
        train_df = clean_df[:split_idx]
        test_df = clean_df[split_idx:]
        train_df_label = pd.DataFrame({"carbon_intensity": pd.Series(train_df['carbon_intensity'])})
        test_df_label = pd.DataFrame({"carbon_intensity": pd.Series(test_df['carbon_intensity'])})
        train_df_features = self.generate_features_df(train_df)
        test_df_features = self.generate_features_df(test_df)
        self.train_df_features = train_df_features
        self.train_df_label = train_df_label
        self.test_df_features = test_df_features
        self.test_df_label = test_df_label


    def train(self, no_existing_model: bool, new_co2_time_series_dataframe: pd.DataFrame) -> None:
        self._co2_time_series_dataframe = self._validate_and_clean_co2_time_series_df(new_co2_time_series_dataframe)
        self._preprocess_co2_time_series_df()
        if no_existing_model:
            self.create_new_model()
            self.model.fit(self.train_df_features, self.train_df_label, 
            early_stopping_rounds=50,
            eval_set=[(self.train_df_features, self.train_df_label), (self.test_df_features, self.test_df_label)],
            eval_metric='mae')
        else:
            # self.model contains most recent model (note that self.load() must be called beforehand)
            self.model.fit(self.train_df_features, self.train_df_label,
            early_stopping_rounds=50,
            eval_set=[(self.train_df_features, self.train_df_label), (self.test_df_features, self.test_df_label)],
            eval_metric='mae', xgb_model=self.model)
        predictions = self.model.predict(self.test_df_features)
        self.recent_results = {"f": predictions, "mae": mean_absolute_error(pd.Series(self.test_df_label['carbon_intensity']), predictions), "mse": mean_squared_error(pd.Series(self.test_df_label['carbon_intensity']), predictions)} 


    def save(self) -> None:
        try:
            self.model.save_model(os.path.join(self.save_location, self.filename))
        except:
            raise Exception("could not save model") 


    def load(self) -> None:
        self.model = xgb.XGBRegressor()
        try:
            self.model.load_model(os.path.join(self.save_location, self.filename))
        except:
            raise Exception("could not load model")


    def predict(self, desired_predictions: pd.DataFrame) -> pd.DataFrame:
        # desired predictions only require a column of datetime (name: Datetime)
        if "Datetime" not in desired_predictions.columns:
            raise Exception("no 'Datetime' column provided")

        desired_predictions.dropna()
        predictions = pd.DataFrame()
        results = pd.DataFrame(columns=["forecast"])
        predictions.set_index(pd.Series(desired_predictions['Datetime']), inplace=True)
        predictions.index = pd.to_datetime(predictions.index)
        finalized_predictions = self.generate_features_df(predictions)
        results["forecast"] = self.model.predict(finalized_predictions)
        return results