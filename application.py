# Import libraries
import numpy as np
from flask import Flask, request, jsonify
from deeppavlov import build_model, configs

app = Flask(__name__)
# Load the model
model = build_model(configs.squad.squad, download=False)
@app.route('/ques',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.

    print(data['passage'])
    print(data['question'])
    prediction = model([data['passage']],[data['question']])
    answer={}
    answer['result']=prediction[0]
    answer['starting'] = prediction[1]
    answer['logitvalue'] = prediction[2]
    # Take the first value of prediction

    return jsonify(answer)
if __name__ == '__main__':
    app.run(port=5000, debug=True)
