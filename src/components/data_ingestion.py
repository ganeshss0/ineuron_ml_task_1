import os, sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from dataclasses import dataclass


# Data Ingesttion Process
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

@dataclass
class DataIngestion:
    config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method Start')
        try:
            data = pd.read_csv(os.path.join('notebooks', 'data', 'finalTrain.csv'))
            logging.info('PASS Dataset Load')

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok = True)
            data.to_csv(self.config.raw_data_path, index = False)
            logging.info(f'Raw File Saved at {self.config.raw_data_path}')

            # Splitting Data in training and testing
            train_data, test_data = tts(data, test_size = 0.3, random_state = 7)
            logging.info('PASS Data Split')

            # Saving Train Test Data
            train_data.to_csv(self.config.train_data_path, index = False, header = True)
            logging.info(f'Train Split Saved at {self.config.raw_data_path}')

            test_data.to_csv(self.config.test_data_path, index = False, header = True)
            logging.info(f'Test Split Saved at {self.config.raw_data_path}')

            logging.info('PASS Data Ingestion')

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        
        except Exception as e:
            logging.warning('Exception occured at Data Ingestion Stage')
            logging.warning(e)
            raise CustomException(e, sys)



from src.components.data_transformation import DataTransform
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    start = DataIngestion()
    train_path, test_path = start.initiate_data_ingestion()
    print(train_path)
    print(test_path)
    data_transform = DataTransform()
    train_data, test_data, preprocessor = data_transform.initiate_data_transform(train_path, test_path)
    model_trainer = ModelTrainer()
    model_path = model_trainer.initiate_model_training(train_data, test_data)
    print('Working Fine')