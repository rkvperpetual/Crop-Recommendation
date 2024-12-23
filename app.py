from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, encoder, and scaler
with open('rnd_clf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect input data from the form
            n = float(request.form.get("n"))
            p = float(request.form.get("p"))
            k = float(request.form.get("k"))
            temperature = float(request.form.get("temperature"))
            humidity = float(request.form.get("humidity"))
            ph = float(request.form.get("ph"))
            rainfall = float(request.form.get("rainfall"))

            # Prepare input data as a DataFrame
            input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                                      columns=["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"])
            input_data = np.array(input_data)
            
            # Preprocess the input data
            scaled_features = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(scaled_features)

            # Decode the prediction
            crop_recommendation = encoder.inverse_transform(prediction)[0]

            return render_template("index.html", result=crop_recommendation)

        except Exception as e:
            return render_template("index.html", result=f"Error: {str(e)}")

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
