from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model
with open("salary_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()

# route for homepage
@app.route("/")
def index():
    return render_template("index.html") 

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
