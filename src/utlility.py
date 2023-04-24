import os, sys
from src.exception import CustomException
from src.logger import logging
import pickle
import numpy as np
import redis
from dataclasses import dataclass

# Radius of Earth (km)
Earth_Radius = 6371

def saveObject(file_path: str, obj: object) -> None:
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok = True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f'FAIL Save Object at {file_path}')

    except Exception as e:
        logging.error(f'FAIL Save Object at {file_path}')
        logging.error(e)
        raise CustomException(e, sys)
    

def format_24_hour(Time: str) -> object:
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

def globe_distance(data: 'DataFrame', x1: str, y1: str, x2: str, y2: str) -> float:

    """Return the distance between (x1, y1) and (x2, y2), Where (x1, y1) are latitude and longitude of first location and 
    (x2, y2) are latitude and longitude of second location."""

    x1, y1 = np.abs(data[x1]), np.abs(data[y1])
    x2, y2 = np.abs(data[x2]), np.abs(data[y2])
    
    lat_diff = degree_radian(x2 - x1) / 2
    lon_diff = degree_radian(y2 - y1) / 2
    d = np.square(np.sin(lat_diff)) + np.cos(degree_radian(x1)) * np.cos(degree_radian(x2)) * np.square(np.sin(lon_diff))
    D = 2 * Earth_Radius * np.arcsin(np.sqrt(d))
    return np.round(D, 2)

def fetch_redis(connection, key, name = 'default'):
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