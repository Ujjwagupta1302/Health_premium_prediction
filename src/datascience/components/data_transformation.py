import os
from src.datascience import logger
from sklearn.model_selection import train_test_split
from src.datascience.entity.config_entity import (DataIngestionconfig, DataValidationConfig,DataTransformationConfig)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transforming_data(self):
        df = pd.read_csv(self.config.data_path) 

        ## Note-> to go through extensive EDA look at the python 
        # notebook named '03_data_transformation_eda.ipynb' int the 
        # same directory as that of this notebook.
        df.columns = df.columns.str.replace(' ', '_').str.lower() # replacing the " " in cloumn names to '_'
        # then converting them to lower case string name 

        # Dropping the null values 
        df.dropna(inplace=True)

        # No duplicare rows present 
        df.drop_duplicates(inplace=True)

        # converting the -ve dependant values to +ve 
        df['number_of_dependants'] = abs(df['number_of_dependants'])

        # treating outlier values of age and income 
        df_1 = df[df.age<=100].copy() # removing age greater than 100

        df_2 = df_1[df_1['income_lakhs']<=100] # removing income greater than 1 cr 

        # Categorical data modification on the smoking_status of our data.
        df_2['smoking_status'].replace({
            'Smoking=0':'No Smoking',
            'Does Not Smoke': 'No Smoking',
            'Not Smoking': 'No Smoking'
        }, inplace = True)

        risk_scores = {
            "diabetes": 6,
            "heart disease": 8,
            "high blood pressure":6,
            "thyroid": 5,
            "no disease": 0,
            "none":0
        }

        df_2[['disease1', 'disease2']] = df_2['medical_history'].str.split(" & ", expand=True).apply(lambda x: x.str.lower())
        df_2['disease1'].fillna('none', inplace=True)
        df_2['disease2'].fillna('none', inplace=True)
        df_2['total_risk_score'] = 0

        for disease in ['disease1', 'disease2']:
            df_2['total_risk_score'] += df_2[disease].map(risk_scores)

        # Normalize the risk score to a range of 0 to 1
        max_score = df_2['total_risk_score'].max()
        min_score = df_2['total_risk_score'].min()
        df_2['normalized_risk_score'] = (df_2['total_risk_score'] - min_score) / (max_score - min_score)

        df_2['insurance_plan'] = df_2['insurance_plan'].map({'Bronze': 1, 'Silver': 2, 'Gold': 3})

        df_2['income_level'] = df_2['income_level'].map({'<10L':1, '10L - 25L': 2, '25L - 40L':3, '> 40L':4})

        nominal_cols = ['gender', 'region', 'marital_status', 'bmi_category', 'smoking_status', 'employment_status']
        df_3 = pd.get_dummies(df_2, columns=nominal_cols, drop_first=True, dtype=int)

        df_4 = df_3.drop(['medical_history','disease1', 'disease2', 'total_risk_score',], axis=1)

        cols_to_scale = ['age','number_of_dependants','income_level', 'income_lakhs', 'insurance_plan']
        scaler = MinMaxScaler()

        df_4[cols_to_scale] = scaler.fit_transform(df_4[cols_to_scale])

        df_4.drop('income_level', axis='columns', inplace=True)

        train, test = train_test_split(df_4)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)


