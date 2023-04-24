import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utlility import saveObject, evalute_model

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

@dataclass
class ModelTrainer:
    config = ModelTrainerConfig()

    def initiate_model_training(self, train_data: np.array, test_data: np.array) -> str:
        try:
            
            logging.info('Splitting Dependent and Independent Variable Train-Test Data')

            X_train, X_test, y_train, y_test = (
                train_data[:, :-1],
                test_data[:, :-1],
                train_data[:, -1],
                test_data[:, -1]
            )

            alphas = [1e-10, 1e-5, 1e-2, 1e-1, 0.5, 1, 2, 3, 5, 10, 20, 30, 40, 50]
            models = {
                'Lasso': LassoCV(
                            alphas = alphas, 
                            cv = 5,
                            ),
                'Ridge': RidgeCV(
                            alphas = alphas, 
                            cv = 5, 
                            ),
                'ElasticNet': ElasticNetCV(
                                alphas = alphas, 
                                cv = 5, 
                                )
            }
        
            
            model_report = {}
                
            model_report = evalute_model(
                                    X_train,
                                    X_test,
                                    y_train,
                                    y_test,
                                    models
                                        )

            logging.info(f'Model Report: \n{model_report}')
            print('Model Reports:\n', model_report, '\n\n')


            best_model_name = model_report.sort_values(by = 'R2Score', ascending = False).iloc[0,0]
            print('Best Model:\n', model_report[model_report.ModelName == best_model_name], '\n\n')
            best_model = models[best_model_name]

            saveObject(
                file_path = self.config.trained_model_file_path,
                obj = best_model
            )
            return self.config.trained_model_file_path

        except Exception as e:
            logging.error('FAILED Model Training')
            logging.error(e)
            raise CustomException(e, sys)