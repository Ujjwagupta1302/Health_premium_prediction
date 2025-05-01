import os
import pandas as pd

from src.datascience import logger
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import joblib
from src.datascience.entity.config_entity import (DataIngestionconfig, DataValidationConfig,ModelTrainerConfig)
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self,var:str,accuracy:float):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column.lower()], axis=1)
        test_x = test_data.drop([self.config.target_column.lower()], axis=1)
        train_y = train_data[[self.config.target_column.lower()]]
        test_y = test_data[[self.config.target_column.lower()]]

        if var == 'ElasticNet' :
            model = ElasticNet() 
            clf = GridSearchCV(model, self.config.params_grid,cv=5, scoring='r2',return_train_score=False)
            clf.fit(train_x, train_y.values.ravel()) 
            curr_accuracy = clf.score(test_x, test_y.values.ravel()) 
            if(curr_accuracy>accuracy) :
                accuracy = curr_accuracy 
                joblib.dump(clf, os.path.join(self.config.root_dir, self.config.model_name))

        else :
            model = RandomForestRegressor()
            clf = GridSearchCV(model, self.config.params_grid, cv=5, scoring='r2', return_train_score=False)
            clf.fit(train_x, train_y.values.ravel())
            curr_accuracy = clf.score(test_x, test_y.values.ravel()) 
            if(curr_accuracy>accuracy) :
                accuracy = curr_accuracy 
                joblib.dump(clf, os.path.join(self.config.root_dir, self.config.model_name))

        return accuracy 