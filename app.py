from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model
with open("salary_model.pkl", "rb") as f:
    model = pickle.load(f)

#  accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    salary = model.predict(input_df)[0]
    return jsonify({"salary": round(salary, 2)})

@app.route("/accuracy", methods=["GET"])
def get_accuracy():
    return jsonify({"accuracy": accuracy})

if __name__ == "__main__":
    app.run(debug=True)
