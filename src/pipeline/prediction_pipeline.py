import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.utlility import loadObject, add_time_feature, globe_distance
import pandas as pd
 

@dataclass
class PredictPipeline:

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.joblib')
            model_path = os.path.join('artifacts', 'model.joblib')

            preprocessor = loadObject(preprocessor_path)
            model = loadObject(model_path)

            scaled_data = preprocessor.transform(features)

            prediction = model.predict(scaled_data)
            logging.info('Successful Predition')
            return prediction
        
        except Exception as e:
            logging.error('FAILED Prediction')
            raise CustomException(e, sys)
        
@dataclass
class CustomData:
    Delivery_person_ID: str
    Delivery_person_Age: float
    Delivery_person_Ratings: float
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weather_conditions: str
    Road_traffic_density: str
    Vehicle_condition: str
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: float
    Festival: str
    City: str

    def get_data_as_dataframe(self):
        try:
            
            col_names = ['Delivery_person_ID', 'Delivery_person_Age', 'Delivery_person_Ratings', 
                         'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 
                         'Delivery_location_longitude', 'Date_Order', 'Time_Orderd', 'Time_Order_picked', 'Weather_conditions', 'Road_traffic_density',
                         'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City']
            
            custom_data_input = [[self.Delivery_person_ID, self.Delivery_person_Age, self.Delivery_person_Ratings, self.Restaurant_latitude, 
                                  self.Restaurant_longitude, self.Delivery_location_latitude, self.Delivery_location_longitude, self.Order_Date, 
                                  self.Time_Orderd, self.Time_Order_picked, self.Weather_conditions, self.Road_traffic_density, self.Vehicle_condition,
                                  self.Type_of_order, self.Type_of_vehicle, self.multiple_deliveries, self.Festival, self.City]]


            data = pd.DataFrame(custom_data_input, columns = col_names)

            # adding columns
            add_time_feature(data)

            # adding distance column
            data['distance_rest_deliv'] = globe_distance(
                data = data, 
                x1 = 'Restaurant_latitude', 
                y1 = 'Restaurant_longitude', 
                x2 = 'Delivery_location_latitude', 
                y2 = 'Delivery_location_longitude'
                )

            logging.info('DataFrame Gathered')
            return data
        except Exception as e:
            logging.error('FAILED DataFrame Gathered')
            logging.error(e)
            raise CustomException(e, sys)
    
# if __name__ == '__main__':
#     data = CustomData('INDORES13DEL02',32.0,4.9,22.745049,75.892471,22.825049,75.972471,
#                       '02-03-2022','23:20','23:30','Cloudy','Low',0,'Meal','motorcycle',2.0,'No','Metropolitian')
#     df = data.get_data_as_dataframe()
#     predict_pipeline = PredictPipeline()
#     res = predict_pipeline.predict(df)
#     print('Delivery Time is', res)
