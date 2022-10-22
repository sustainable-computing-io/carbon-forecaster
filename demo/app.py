from flask import Flask, request, json, Response
from forecaster import forecast_model

app = Flask(__name__)

@app.route('/forecaster', methods=['GET'])
def forecast_region():
    args = request.args
    model_type = args.get("model_type", default="ARIMAModel", type=str)
    region = args.get("region", default="GB", type=str)
    steps_into_future = args.get("steps", default=1, type=int)
    model_res = forecast_model(model_type, region, steps_into_future)
    retrieved_json = model_res.to_json(orient="records")
    response = app.response_class(
        response=retrieved_json,
        status=200,
        mimetype='application/json'
        )
    print(model_res)
    return response


if __name__ == "__main__":
    app.run(debug=True)