import pickle
from flask import Flask , request , url_for , render_template , app , jsonify
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)
## Load The model 
model = pickle.load(open("regression_model.pkl" , 'rb'))
scaler = pickle.load(open('scaling.pkl' , 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api' , method = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1 , -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1 , -1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])
@app.route('/predict' , method = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1 , -1))
    print(final_input)
    ouput = model.predict(final_input)[0]
    return render_template("home.html" , prediction_text = "The HOuse price  prediction is {}"  .format(ouput) )


if __name__== "__main__":
    app.run(debug=True)
