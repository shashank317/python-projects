import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)  # ← renamed here
model = pickle.load(open('model.pkl', "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)
    return render_template('index.html', prediction_text='Predicted Crop is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
