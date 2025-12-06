from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model pipeline
with open('model_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

# Default values for the rest of the features
default_values = {
    'KIDSDRIV': 0,
    'HOMEKIDS': 0,
    'YOJ': 11,
    'PARENT1': 0,
    'MSTATUS': 'Yes',
    'GENDER': 'z_F',
    'EDUCATION': 'z_High School',
    'OCCUPATION': 'z_Blue Collar',
    'TRAVTIME': 33,
    'CAR_USE': 'Private',
    'BLUEBOOK': 1500,
    'TIF': 4,
    'CAR_TYPE': 'z_SUV',
    'RED_CAR': 0,
    'CLM_FREQ': 0,
    'REVOKED': 0,
    'claim_income_ratio': None
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/guidelines')
def guidelines():
    return render_template('guidelines.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict_result', methods=['POST'])
def predict_result():
    try:
        # ✅ Collect user inputs
        user_input = {
            'AGE': float(request.form['AGE']),
            'INCOME': float(request.form['INCOME']),
            'HOME_VAL': float(request.form['HOME_VAL']),
            'CAR_AGE': float(request.form['CAR_AGE']),
            'OLDCLAIM': float(request.form['OLDCLAIM']),
            'CLM_AMT': float(request.form['CLM_AMT']),
            'MVR_PTS': float(request.form['MVR_PTS'])
        }

        # ✅ Merge with default values
        final_input = {**default_values, **user_input}

        # ✅ Convert to DataFrame
        final_df = pd.DataFrame([final_input])

        # ✅ Predict
        prediction = model.predict(final_df)[0]

        result = "🚨 Fraudulent Claim Detected!" if prediction == 1 else "✅ Genuine Claim (No Fraud)"
        color = "danger" if prediction == 1 else "success"

        return render_template('predict.html', result=result, color=color)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
