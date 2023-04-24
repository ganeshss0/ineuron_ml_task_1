import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utlility import saveObject, format_24_hour, globe_distance, fetch_redis, redis_connect

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass




@dataclass
class DataTransformConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')



@dataclass
class DataTransform:
    config = DataTransformConfig()
    r = redis_connect(
        host = 'delivery-time-info.redis.cache.windows.net',
        port = 6380,
        db =0,
        password = 'g1ohvaqFaRceQYOUFfG14fPOh6rp1JuoNAzCaHTjM9s=',
        ssl = True,
        decode_responses = True
    )
    
    def add_time_feature(self, data) -> None:
        
        # Appling the format_24_hour function on both Time_Orderd and Time_Order_picked columns
        data['Time_Orderd'] = data.Time_Orderd.apply(format_24_hour)
        data['Time_Order_picked'] = data.Time_Order_picked.apply(format_24_hour)

        order = data[(data.Time_Order_picked.notna() & data.Time_Orderd.notna())]


        # Converting the Time_Orderd into DateTime object
        order_time = pd.to_datetime(order['Time_Orderd'])

        # Converting the Time_Order_picked to DateTime object
        order_picked = pd.to_datetime(order['Time_Order_picked'])

        # Median difference between order picked and order time(seconds)
        median_order_pick_time = (order_picked - order_time).dt.seconds.median()
        
        # Selecting only those columns where Time_Ordered is NaN and Time_Order_picked is not NaN
        order = data.loc[data.Time_Orderd.isna() & data.Time_Order_picked.notna()]

        # Converting into DateTime object
        order_picked = pd.to_datetime(order['Time_Order_picked'])

        # Here subtracting Median Order Picking Time (600 seconds) from Order Picked Time
        data['Time_Orderd'].fillna(
            value = (order_picked - pd.Timedelta(seconds = median_order_pick_time)).dt.strftime('%H:%M'), 
            inplace = True
            )

        data['order_hour'] = data['Time_Orderd'].str.split(':', expand = True)[0].astype(float)
    
    def distance_from_locations(self, data):
        data['distance_rest_deliv'] = globe_distance(
                data = data, 
                x1 = 'Restaurant_latitude', 
                y1 = 'Restaurant_longitude', 
                x2 = 'Delivery_location_latitude', 
                y2 = 'Delivery_location_longitude'
                )
        data.drop([
        'Restaurant_latitude', 
        'Restaurant_longitude', 
        'Delivery_location_latitude', 
        'Delivery_location_longitude'], axis = 1, inplace = True)

    def build_pipeline(self):
        try:
            logging.info('Build Pipeline Initiated')

            
            # Numerical Pipeline

            nums_pipe = Pipeline(
                steps = (
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaling', StandardScaler())
                )
            )

            logging.info('Numerical Pipeline Created')
            # Categorical Pipeline

            categories = [
                fetch_redis(self.r, 'categories-personID'),
                fetch_redis(self.r, 'categories-weather'),
                fetch_redis(self.r, 'categories-traffic'),
                fetch_redis(self.r, 'categories-order'),
                fetch_redis(self.r, 'categories-vehicle'),
                fetch_redis(self.r, 'categories-festival'),
                fetch_redis(self.r, 'categories-city'),
                ]
            
            cats_pipe = Pipeline(
                steps = (
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories = categories)),
                ('scaler', StandardScaler())
                )
            )

            logging.info('Categorical Pipeline Created')

            preprocessor = ColumnTransformer(
                transformers = [
                ('Num', nums_pipe, fetch_redis(self.r, 'numerical')),
                ('Cat', cats_pipe, fetch_redis(self.r, 'categorical'))
                ]
            )
            logging.info('Build Pipeline Successful')

            return preprocessor

        except Exception as e:
            logging.error('Build Pipeline Failed')
            logging.error(e)
            raise CustomException(e, sys)
    

    def initiate_data_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            

            logging.info('Train Test Data Loaded Succesful')
            logging.info(f'Train DataFrame: \n{train_df.head(3).to_string()}')
            logging.info(f'Test DataFrame: \n{test_df.head(3).to_string()}')
            
            target_col_name = 'Time_taken (min)'
            drop_columns = ['ID']

            features_train_data = train_df.drop(columns = drop_columns, axis = 1)
            target_train_data = train_df[target_col_name]

            features_test_data = test_df.drop(columns = drop_columns, axis = 1)
            target_test_data = test_df[target_col_name]


            preprocessor = self.build_pipeline()
            logging.info('Pipeline Loaded Successful')

            self.add_time_feature(features_train_data)
            self.add_time_feature(features_test_data)


            self.distance_from_locations(features_train_data)
            self.distance_from_locations(features_test_data)

            preprocessor.fit(features_train_data)

            transform_feature_train_data = preprocessor.transform(features_train_data)
            transform_feature_test_data = preprocessor.transform(features_test_data)

            logging.info('Data Transformation Successful')

            train_data = np.c_[transform_feature_train_data, np.array(target_train_data)]
            test_data = np.c_[transform_feature_test_data, np.array(target_test_data)]

            saveObject(
                file_path = self.config.preprocessor_obj_path,
                obj = preprocessor
            )

            return (
                train_data,
                test_data,
                self.config.preprocessor_obj_path
            )

        except Exception as e:
            logging.error('Data Transform Failed')
            logging.error(e)
            raise CustomException(e, sys)