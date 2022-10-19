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
    new_df = new_uk_df.tail(1200)
    first_df = new_df.iloc[:200,:]
    second_df = new_df.iloc[200:400,:]
    third_df = new_df.iloc[400:600,:]
    fourth_df = new_df.iloc[600:800,:]
    fifth_df = new_df.iloc[800:1000,:]
    sixth_df = new_df.iloc[1000:,:]
    #print(first_df.head())
    #print(first_df.tail())
    #print("\n")
    #print(second_df.head())
    #print(second_df.tail())
    #print("\n")
    #print(third_df.head())
    #print(third_df.tail())
    #print("\n")
    #print(fourth_df.head())
    #print(fourth_df.tail())
    #print("\n")
    #print(fifth_df.head())
    #print(fifth_df.tail())
    #print("\n")
    #print(sixth_df.head())
    #print(sixth_df.tail())
    #print("\n")

    #UK_Arima = produce_initial_generic_forecast_model(first_df, ARIMAModel("GB"))
    UK_Arima = GenericModelTrainingPipelinePerRegion(ARIMAModel("GB"))
    UK_Arima.train_model(sixth_df, False, False)

#done