from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("models/credit_scoring_model.pkl", "rb"))
scaler = pickle.load(open("models/credit_scaler.pkl", "rb"))



@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/analyzer")
def analyzer_page():
    return render_template("index.html")

@app.route('/features')
def features_page():
    return render_template('features.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        numeric_data = [
            float(data["age"]),
            float(data["income"]),
            float(data["salary"]),
            float(data["loan_count"]),
            float(data["emi"]),
            float(data["debt"]),
        ]
        numeric_scaled = scaler.transform([numeric_data])[0]

      
        credit_mix_map = {"Bad": 0, "Standard": 1, "Good": 2}
        payment_min_map = {"No": 0, "Yes": 1, "NM": 2}

        credit_mix = credit_mix_map.get(data["credit_mix"], 1)
        payment_min = payment_min_map.get(data["payment_min"], 2)

        final_input = np.concatenate([numeric_scaled, [credit_mix, payment_min]])

        # Prediction
        prediction = model.predict([final_input])[0]
        result = "High Risk" if prediction == 1 else "Low Risk"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
