from forecast_pipeline import GenericModelTrainingPipelinePerRegion
from forecast_models import GenericCarbonIntensityForecastModel, NeuralProphetModel, ARIMAModel
import os
import pandas as pd


def produce_initial_generic_forecast_model(dataframe: pd.DataFrame, model: GenericCarbonIntensityForecastModel):
    # produce pretrained models using mock data
    generic_model_ref = GenericModelTrainingPipelinePerRegion(model)
    generic_model_ref.train_model(dataframe, True, False)

    return generic_model_ref

if __name__ == "__main__":

    # UK Co2 Initial Dataset
    UK_Initial_Dataset_url = os.path.join(os.path.dirname(__file__), "pre_trained/initial_datasets/GB_fuel_ckan.csv")
    uk_df = pd.read_csv(UK_Initial_Dataset_url)
    new_uk_df = pd.DataFrame(columns=["datetime", "carbon_intensity"])
    new_uk_df["datetime"] = uk_df["DATETIME"]
    new_uk_df["carbon_intensity"] = uk_df["CARBON_INTENSITY"]
    
    UK_Arima = produce_initial_generic_forecast_model(new_uk_df.tail(200), ARIMAModel("GB"))
