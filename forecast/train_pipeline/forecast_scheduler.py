import pandas as pd
from forecast_pipeline import GenericModelTrainingPipelinePerRegion, XGBRegressorModelTrainingPipelinePerRegion
from forecast_models import NeuralProphetModel, ARIMAModel
from typing import Dict
import requests
import os
import time

@staticmethod
def scrape_prom_metrics(query, interval, endpoint):
    query_str= 'query={}[{}]'.format(query, interval)
    return requests.get(url = endpoint, params = query_str).json()

@staticmethod
def retrieve_co2_intensity_data(co2_endpoint, co2_query, co2_interval) -> Dict[str, pd.DataFrame]:
    co2_metrics_json = scrape_prom_metrics(co2_query, co2_interval, co2_endpoint)        
    result = co2_metrics_json['data']['result']
    region_to_prom_metrics = {}

    for x in range(len(result)):
        metrics = result[x]['metric']
        region = str(metrics['region'])
        # check if string is being properly converted
        # need to retrieve prometheus time stamp instead
        datetime_data = str(metrics['datetime'])
        carbon_intensity_data = float(metrics['carbon_intensity'])
        if region not in region_to_prom_metrics:
            region_to_prom_metrics[region] = {"datetime": [], "carbon_intensity_data": []}
        region_to_prom_metrics[region]['datetime'].append(datetime_data)
        region_to_prom_metrics[region]['carbon_intensity'].append(carbon_intensity_data)

    # create dataframes of regions
    carbon_dataframes_all_regions = {}
    for region in region_to_prom_metrics.keys():
        co2_carbon_data_per_region = pd.DataFrame(columns=["datetime", "carbon_intensity"])
        co2_carbon_data_per_region['datetime'] = region_to_prom_metrics[region]['datetime']
        co2_carbon_data_per_region['carbon_intensity'] = region_to_prom_metrics[region]['carbon_intensity']
        carbon_dataframes_all_regions[region] = co2_carbon_data_per_region
    
    return carbon_dataframes_all_regions


def multi_region_prom_scraper_and_training(query, interval, endpoint):
    start = time.time()

    regions_to_train = retrieve_co2_intensity_data(query, interval, endpoint)
    # perform initial training here
    region_to_pipelines = {}
    for region in regions_to_train.keys():
        if region not in region_to_pipelines:
            region_to_pipelines[region] = {"NeuralProphet": None, "ARIMA": None}
        neural_prophet_model = NeuralProphetModel(region)
        neural_prophet_training_pipeline = GenericModelTrainingPipelinePerRegion(neural_prophet_model)
        neural_prophet_training_pipeline.train_model(regions_to_train[region], True)
        region_to_pipelines[region]["NeuralProphet"] = neural_prophet_training_pipeline
        ARIMA_model = ARIMAModel(region)
        ARIMA_training_pipeline = GenericModelTrainingPipelinePerRegion(ARIMA_model)
        ARIMA_training_pipeline.train_model(regions_to_train[region], True)
        
        region_to_pipelines[region]["ARIMA"] = ARIMA_training_pipeline


    end = time.time()
    while True:
        time_lapsed = end - start
        if time_lapsed > interval_sec:
            start = time.time()
            regions_to_train = retrieve_co2_intensity_data(query, '{}s'.format(int(time_lapsed)), endpoint)
            # perform incremental training here
            for region in regions_to_train.keys():
                region_to_pipelines[region]["ARIMA"].train_model(regions_to_train[region], False)
                region_to_pipelines[region]["NeuralProphet"].train_model(regions_to_train[region], False)
        else: 
            time.sleep(int(interval_sec - time_lapsed))
        end = time.time()


if __name__ == "__main__":
    endpoint = os.getenv('PROMETHEUS_CO2_ENDPOINT', default='http://localhost:9090/api/v1/query')
    query = os.getenv('PROMETHEUS_CO2_QUERY', default='carbon_intensity')
    interval_sec = os.getenv('PROMETHEUS_QUERY_INTERVAL', default=10)
    interval = '{}s'.format(interval_sec)


    # continue to update models given new data

    start = time.time()

    regions_to_train = retrieve_co2_intensity_data(query, interval, endpoint)

    end = time.time()
    while True:
        time_lapsed = end - start
        if time_lapsed > interval_sec:
            start = time.time()
            regions_to_train = retrieve_co2_intensity_data(query, '{}s'.format(int(time_lapsed)), endpoint)
        else: 
            time.sleep(int(interval_sec - time_lapsed))
        end = time.time()