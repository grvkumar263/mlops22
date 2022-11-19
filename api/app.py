from flask import Flask
from flask import request
from joblib import load

import sys
sys.path.append('.')
import os

#from utils import preprocess_digits

#from sklearn import datasets
#digits = datasets.load_digits()
#data, label = preprocess_digits(digits)




app = Flask(__name__)

@app.route('/')
def hello():
    return "<b> Hello world </b>"

@app.route('/sum', methods = ['POST'])
def sum(x,y):
    print(request.json)
    x = request.json['x']
    y = request.json['y']
    z = x+y
    return {'sum': z}

model_path = "/exp/svm_gamma=0.001_C=0.7.joblib"
model = load(model_path)


@app.route('/predict', methods = ['POST'])
def predict_digit():
    image = request.json['image']
    
    predicted = model.predict([image])
    
    return {"predicted" : int(predicted[0])}


@app.route('/compareimages', methods = ['POST'])
def compare_images():
    image1 = request.json['image1']
    image2 = request.json['image2']
    
    predict1 = model.predict([image1])
    predict2 = model.predict([image2])
    if int(predict1[0]) == int(predict2[0]):
        return  {"result" :"Same numbers"}
    else:
        return {"result" :"Different numbers"}



#export FLASK_APP=api/app.py
#flask run
#curl 'https://127.0.0.1:5000/sum' -X POST -H 'Content-Type: application/json' -d '{"x":5, "y":4}'


#curl http://127.0.0.1:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"image": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}'

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)