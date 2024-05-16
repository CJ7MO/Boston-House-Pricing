import pickle
from flask import Flask,request,app,jsonify,render_template, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
lr_model = pickle.load(open('reg_model.pkl','rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    df = request.json['data']
    print(df)
    print(np.array(list(df.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(df.values())).reshape(1,-1))
    output = lr_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    df = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(df).reshape(1, -1))
    print(final_input)
    output = lr_model.predict(final_input)[0]
    return render_template("home.html", prediction_text= "The House price prediction is: {:.2f}".format(output))

if __name__=="__main__":
    app.run(host="0.0.0.0", port = 8080, debug=True)
    #app.run(host="0.0.0.0", port = 8080)
