from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
from AI_model import fish_weight_predictor

app = Flask(__name__)

# fish_weight_prediction_model = os.path.join(os.path.dirname(__file__), 'fish_weight_predictor.pkl')

# if not os.path.isfile(fish_weight_prediction_model):
#     fish_weight_predictor()

# Load the trained model
model = joblib.load('fish_weight_predictor.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    species = request.form.get('species')
    length1 = request.form.get('length1')
    length2 = request.form.get('length2')
    length3 = request.form.get('length3')
    height = request.form.get('height')
    width = request.form.get('width')
    
    data = pd.DataFrame({
        'Species': [species], 
        'Length1': [length1], 
        'Length2': [length2], 
        'Length3': [length3], 
        'Height': [height], 
        'Width': [width]
        })

    prediction = model.predict(data)

    return render_template('prediction.html', predicted_fish_weight=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)