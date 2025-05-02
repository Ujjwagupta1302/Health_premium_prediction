from flask import Flask, render_template, request
import os
import numpy as np
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

risk_scores = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0
}

@app.route('/', methods=['GET'])
def homepage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training successful"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Basic numeric fields
            age = float(request.form['age'])
            number_of_dependants = float(request.form['number_of_dependants'])
            income_lakhs = float(request.form['income_lakhs'])
            insurance_plan = float(request.form['insurance_plan'])

            # Disease checkboxes (can select multiple)
            selected_diseases = request.form.getlist('diseases')
            total_score = sum(risk_scores.get(disease.lower(), 0) for disease in selected_diseases)
            normalized_risk_score = (total_score - 0) / (14 - 0)  # Normalize between 0 and 1

            # Dropdowns (all numeric 0 or 1 for one-hot style encoding)
            gender = request.form['gender']
            region = request.form['region']
            marital_status = request.form['marital_status']
            bmi_category = request.form['bmi_category']
            smoking_status = request.form['smoking_status']
            employment_status = request.form['employment_status']

            # One-hot encoded features from dropdowns
            gender_Male = 1 if gender == "Male" else 0

            region_Northwest = 1 if region == "Northwest" else 0
            region_Southeast = 1 if region == "Southeast" else 0
            region_Southwest = 1 if region == "Southwest" else 0

            marital_status_Unmarried = 1 if marital_status == "Unmarried" else 0

            bmi_category_Obesity = 1 if bmi_category == "Obesity" else 0
            bmi_category_Overweight = 1 if bmi_category == "Overweight" else 0
            bmi_category_Underweight = 1 if bmi_category == "Underweight" else 0

            smoking_status_Occasional = 1 if smoking_status == "Occasional" else 0
            smoking_status_Regular = 1 if smoking_status == "Regular" else 0

            employment_status_Salaried = 1 if employment_status == "Salaried" else 0
            employment_status_Self_Employed = 1 if employment_status == "Self-Employed" else 0

            # Final input data
            data = [age, number_of_dependants, income_lakhs, insurance_plan,
                    normalized_risk_score, gender_Male, region_Northwest,
                    region_Southeast, region_Southwest, marital_status_Unmarried,
                    bmi_category_Obesity, bmi_category_Overweight,
                    bmi_category_Underweight, smoking_status_Occasional,
                    smoking_status_Regular, employment_status_Salaried,
                    employment_status_Self_Employed]

            data = np.array(data).reshape(1, -1)  # Automatically deduce shape

            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is:', e)
            return 'Something went wrong during prediction.'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
