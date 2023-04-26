import os
import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform
from src.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    data_transform = DataTransform()
    train_data, test_data, _= data_transform.initiate_data_transform(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_data, test_data)