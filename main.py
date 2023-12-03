from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the saved model using pickle
with open('cat_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


def predict_function(parameters):
    # Convert the input parameters into a DataFrame
    input_data = pd.DataFrame([parameters])

    # Make predictions
    y_pred = model.predict(input_data)

    # Get probability estimates
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)[:, 1]
    else:
        # For models without predict_proba, use decision function for SVM
        probabilities = model.decision_function(input_data)

    # Return prediction and probability
    return int(y_pred[0]), round(max(probabilities) * 100, 2)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)
        stats = data['stats']
        # Assuming your hf_model function is modified to take parameters and return the prediction
        prediction, probability = predict_function(stats)
        return jsonify({'prediction': prediction, 'probability': probability})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


@app.route('/get_ai_response', methods=['POST'])
def get_ai_response():

    # import ipdb;
    # ipdb.set_trace();
    key = os.getenv('OPENAI_API_KEY')
    print(key)

    client = OpenAI(
        api_key=key,
    )

    data = request.get_json()

    user_details = data['user_details']
    user_stats = data['stats']
    prediction_result = data['prediction_result']

    print(user_details, user_stats, prediction_result)

    system_text = '''
Imagine you are a AI medical expert specializing in heart diseases, and you have received a health report for evaluation of a patient. The report includes the following inputs:
Age, Anaemia, Creatinine Phosphokinase Levels, Ejection Fraction, High Blood Pressure, Platelet Count, Serum Creatinine Levels, Serum Sodium Levels, Gender, Diabetes, Smoking, Number of days since last checking

and the Machine Learning model has given the prediction and probability report which indicates chance of heart failure.

Now based on these, your details as a AI Heart Expert Model, give a short but detailed report on user's heart condition, and it should be more of a conversational style. Please include information regarding user's current heart conditions, and helpful tips, precautions personalised to the report. Please add a point regarding taking second opinion from a doctor. Make sure the response is in HTML format and this is a must.
'''
    prompt_text = f'''
User Details: {user_details} 

User Stats:
{user_stats}

Model prediction : {prediction_result}
'''
    print(system_text)
    print(prompt_text)

    prompt_messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_text}
    ]

    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=prompt_messages,
        temperature=0,
        max_tokens=int(4000 - (len(json.dumps(prompt_messages)) / 4)),
    )

    # print(response)

    generated_text = response.choices[0].message.content

    return jsonify({'generated_text': generated_text})


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
