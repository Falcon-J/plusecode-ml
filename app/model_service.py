from flask import Flask, request, jsonify
from ml_pipeline import predict_fatigue

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_fatigue(data)
    return jsonify({"fatigue_score": result})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
