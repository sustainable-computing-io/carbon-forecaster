from flask import Flask, request, json, Response
from train_pipeline.forecast_models import ARIMAModel, NeuralProphetModel

app = Flask(__name__)

@app.route('/forecaster', methods=['GET'])
def forecast_region():
    args = request.args
    model_type = args.get("model_type", default="ARIMAModel", type=str)
    region = args.get("region", default="GB", type=str)
    steps_into_future = args.get("steps", default=1, type=int)
    if model_type == "ARIMAModel":
        model = ARIMAModel(region)
    elif model_type == "NeuralProphetModel":
        model = NeuralProphetModel(region)
    model.load()
    retrieved_json = model.forecast(steps_into_future).to_json(orient="records")
    response = app.response_class(
        response=retrieved_json,
        status=200,
        mimetype='application/json'
        )
    return response


if __name__ == "__main__":
    app.run(debug=True)