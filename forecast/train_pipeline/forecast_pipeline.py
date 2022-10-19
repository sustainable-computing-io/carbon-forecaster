from forecast_models import GenericCarbonIntensityForecastModel, XGBoostRegressorModel
import pandas as pd

class GenericModelTrainingPipelinePerRegion():

    def __init__(self, model: GenericCarbonIntensityForecastModel) -> None:
        self.model = model
        self.region = model.region
        self.step_type = model.step_type

    def train_model(self, co2_carbon_data, first_time, include_test_df):
        if not first_time:
            self.model.load()
        self.model.train(first_time, co2_carbon_data, include_test_df)
        self.model.save()

    def forecast_model(self, steps_into_future=0):
        results = self.model.forecast(steps_into_future)
        print(results)
        return results.to_json(orient="records")


class XGBRegressorModelTrainingPipelinePerRegion():
    def __init__(self, model: XGBoostRegressorModel) -> None:
        self.model = model
        self.region = model.region
        self.step_type = model.step_type

    def train_model(self, co2_carbon_data, first_time):
        if not first_time:
            self.model.load()
        self.model.train(first_time, co2_carbon_data)
        self.model.save()

    def forecast_model(self, list_of_predictions):
        predictions = pd.DataFrame(columns=["Datetime"])
        predictions["Datetime"] = list_of_predictions
        results = self.model.predict(predictions)
        return results.to_json(orient="records")