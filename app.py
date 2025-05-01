from flask import Flask, render_template, request 
import os 
import numpy as np 
import pandas as pd 
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__) 

@app.route('/', methods = ['GET']) ## route to display the homepage
def homepage():
    return render_template("index.html") 

@app.route('/train', methods = ['GET']) 
def training() :
    os.system("python main.py")
    return "Training successful" 


@app.route('/predict', methods = ['POST','GET']) 
def index():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            age = float(request.form['age'])
            number_of_dependants = float(request.form['number_of_dependants'])
            income_lakhs = float(request.form['income_lakhs'])
            insurance_plan = float(request.form['insurance_plan'])
            normalized_risk_score = float(request.form['normalized_risk_score'])
            gender_Male = float(request.form['gender_Male'])
            region_Northwest = float(request.form['region_Northwest'])
            region_Southeast = float(request.form['region_Southeast'])
            region_Southwest = float(request.form['region_Southwest'])
            marital_status_Unmarried = float(request.form['marital_status_Unmarried'])
            bmi_category_Obesity = float(request.form['bmi_category_Obesity'])
            bmi_category_Overweight = float(request.form['bmi_category_Overweight'])
            bmi_category_Underweight = float(request.form['bmi_category_Underweight'])
            smoking_status_Occasional = float(request.form['smoking_status_Occasional'])
            smoking_status_Regular = float(request.form['smoking_status_Regular'])
            employment_status_Salaried = float(request.form['employment_status_Salaried'])
            employment_status_Self_Employed = float(request.form['employment_status_Self_Employed'])

            # Prepare the data for prediction
            data = [age, number_of_dependants, income_lakhs, insurance_plan, normalized_risk_score, gender_Male, 
                    region_Northwest, region_Southeast, region_Southwest, marital_status_Unmarried, bmi_category_Obesity, 
                    bmi_category_Overweight, bmi_category_Underweight, smoking_status_Occasional, smoking_status_Regular, 
                    employment_status_Salaried, employment_status_Self_Employed]
            data = np.array(data).reshape(1, 17)

            # Prediction pipeline
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))
        
        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong'
        
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080)
