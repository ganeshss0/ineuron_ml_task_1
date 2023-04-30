import os, sys
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
from src.pipeline.training_pipeline import start_training
from src.utlility import redis_connect, fetch_redis
from src.logger import logging
from src.exception import CustomException





app = Flask(__name__)
CORS(app)


@app.route('/')
@cross_origin()
def homePage():
    '''Render Home Page'''
    logging.info('HomePage Rendered')
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
@cross_origin()
def predict():
    try:
        form_data = get_form_data()
        
        data = CustomData(*form_data)
        df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(df)[0]
        return render_template('result.html', result = result)

    except Exception as e:
        logging.error('Prediction Failed')
        logging.error(e)
        raise CustomException(e, sys)
    
@app.route('/login', methods=['GET', "POST"])
@cross_origin()
def login():
    if request.method == 'GET':
        return render_template('login.html', message = 'Login')
    else:
        user_name = request.form['email']
        password = request.form['password']
        r = redis_connect(
        host = 'delivery-time-info.redis.cache.windows.net',
        port = 6380,
        db =0,
        password = 'g1ohvaqFaRceQYOUFfG14fPOh6rp1JuoNAzCaHTjM9s=',
        ssl = True,
        decode_responses = True
    )
        if password == fetch_redis(r, 'users', user_name):
            global is_logined
            is_logined = True
            return render_template('train.html')
        else:
            return render_template('login.html', message = 'Invalid Credentials')

is_logined = False

@app.route('/train', methods=['POST'])
@cross_origin()
def train():
    if is_logined:
        url = request.form['dataset-url']
        start_training(url)
        log = sorted(os.listdir(os.path.join('logs')))[-1]
        with open(os.path.join('logs', log)) as log_file:
            logs = log_file.readlines()
        return render_template('train_result.html', logs = logs)
    else:
        return render_template('login.html', message = 'Login')
    
def get_form_data():
    try:
        location = request.form['location']
        resNum = request.form['resNum']
        delNum = request.form['delNum']

        form_data = [
        # delivery person id
        location + 'RES' + resNum + 'DEL' + delNum,

        # delivery person age
        int(request.form['age']),

        # delivery person rating
        float(request.form['rating']),

        # restaurant latitude
        float(request.form['resLat']),

        # restaurant longitude
        float(request.form['resLong']),

        # delivery location latitude
        float(request.form['delLat']),

        # delivery location longitude
        float(request.form['delLong']),

        # order date
        request.form['date'],

        # order time
        request.form['orderTime'],

        # order picked time
        request.form['orderPickTime'],

        # weather conditions
        request.form['weather'],

        # road traffic density
        request.form['traffic'],

        # vehicle conditions
        int(request.form['vehicleCondition']),

        # order type
        request.form['orderType'],

        # Type of Vehicle
        request.form['vehicle'],

        # Multiple Deliveries
        int(request.form['multipleDelivery']),

        # Festival
        request.form['festival'],

        # Type of City
        request.form['city']
        ]
        logging.info('Form Request Success')

        return form_data
    
    except Exception as e:
        logging.error('Form Request Failed')
        logging.error(e)
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    logging.info('Application Started')
    app.run()
