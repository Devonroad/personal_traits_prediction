#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
from flask import Flask, jsonify, request

input_file = 'model.bin'

with open(input_file, 'rb') as f_in:
    wrapper, model = pickle.load(f_in)

app = Flask('MBTI_personality_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    visitor = pd.DataFrame([request.get_json()])

    y_pred = wrapper.predict_proba(visitor, model)[:,1]
    class_idx = wrapper.predict_class(visitor, model)[0]

    # only temporary
    label = {0: 'ENFJ',
             1: 'ENFP',
             2: 'ENTJ',
             3: 'ENTP',
             4: 'ESFJ',
             5: 'ESFP',
             6: 'ESTJ',
             7: 'ESTP',
             8: 'INFJ',
             9: 'INFP',
             10: 'INTJ',
             11: 'INTP',
             12: 'ISFJ',
             13: 'ISFP',
             14: 'ISTJ',
             15: 'ISTP'}

    result = {
        'personality_proba': float(y_pred),
        'personality_predict': str(label[class_idx])
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8185)
