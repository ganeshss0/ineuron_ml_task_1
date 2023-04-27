import os, sys
from src.exception import CustomException
from src.logger import logging
import pickle
import numpy as np
import pandas as pd
import redis
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Any

# Radius of Earth (km)
Earth_Radius = 6371

def saveObject(file_path: str, obj: object) -> None:
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok = True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f'Successful Save Object at {file_path}')

    except Exception as e:
        logging.error(f'FAIL Save Object at {file_path}')
        logging.error(e)
        raise CustomException(e, sys)
    
def loadObject(file_path: str) -> None:
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logging.info(f'Successful Read Object at {file_path}')
        return obj
    
    except Exception as e:
        logging.error(f'FAIL Read Object at {file_path}')
        logging.error(e)
        raise CustomException(e, sys)



def format_24_hour(Time: str) -> Any:
    '''Returns string/NaN, If input is NaN then it return NaN, else based on the conditions.\n
        23:12 -> 23:12\n
        24:00 -> 00:00\n
        10:00:00 -> 10:00\n
        0.422 -> NaN\n
        NaN -> NaN'''
    try:
        # Convert 24:00 into 00:00
        if Time[:2] == '24':
            Time = '00' + Time[2:]
        
        # Convert 10:00:00 into 10:00
        if (n:=Time.count(':')) == 2:
            Time =  Time[:-3]

        # Convert decimal into NaN
        if n == 0:
            Time =  np.NaN
        
        return Time
    
    except:
        return Time
    

# convert degree into radian
degree_radian = lambda x: x * (np.pi/180)

def globe_distance(data: pd.DataFrame, x1: str, y1: str, x2: str, y2: str) -> float:

    """Return the distance between (x1, y1) and (x2, y2), Where (x1, y1) are latitude and longitude of first location and 
    (x2, y2) are latitude and longitude of second location."""

    x1, y1 = np.abs(data[x1]), np.abs(data[y1])
    x2, y2 = np.abs(data[x2]), np.abs(data[y2])
    
    lat_diff = degree_radian(x2 - x1) / 2
    lon_diff = degree_radian(y2 - y1) / 2
    d = np.square(np.sin(lat_diff)) + np.cos(degree_radian(x1)) * np.cos(degree_radian(x2)) * np.square(np.sin(lon_diff))
    D = 2 * Earth_Radius * np.arcsin(np.sqrt(d))
    return np.round(D, 2)

def fetch_redis(connection, key, name = 'default') -> Any:
    try:
        logging.info('Try Fetch Data Redis Cloud')
        if key == 'users':
            data = connection.hget('users', name)
        else:
            data = connection.lrange(key, 0, -1)

        logging.info('Successful Fetch Data Redis Cloud')
        return data
    
    except Exception as e:
        logging.error('Failed Fetch Data Redis Cloud')
        logging.error(e)
        raise CustomException(e, sys)
    


def redis_connect(host: str, port: int, password: str, db: int, ssl: bool, **kwargs) -> redis.StrictRedis:
    try: 
        cnct = redis.StrictRedis(
            host = host,
            port = port,
            db = db,
            password = password,
            ssl = ssl,
            **kwargs
        )
        if cnct.ping():
            logging.info('Connection Successful Redis Cloud')
        return cnct
    except Exception as e:
        logging.error('Connection Failed Redis Cloud')
        logging.error(e)
        raise CustomException(e, sys)
    

def evalute_model(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, models: dict) -> pd.DataFrame:
    try:
        report = []

        for model in models:
            MODEL = models[model]

            # Model Training
            MODEL.fit(X_train, y_train)

            # Predict Test Data
            y_pred = MODEL.predict(X_test)

            # Evaluation

            report.append([
                model,
                mean_absolute_error(y_test, y_pred),
                r2_score(y_test, y_pred),
                np.sqrt(mean_squared_error(y_test, y_pred))
            ])

        return pd.DataFrame(report, columns = ['ModelName', 'MAE', 'R2Score', 'RMSE'])

    except Exception as e:
        logging.error('FAILED to Train Model')
        logging.error(e)
        raise CustomException(e, sys)



def add_time_feature(data: pd.DataFrame) -> None:
    
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
