from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd


model_filename = 'house_price_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        beds = float(request.form['beds'])
        baths = float(request.form['baths'])
        size = float(request.form['size'])
        zip_code = int(request.form['zip_code'])
        
        features = np.array([[beds, baths, size, zip_code]])
        
        input_df = pd.DataFrame(features, columns=['beds', 'baths', 'size', 'zip_code'])

        predicted_price = model.predict(input_df)[0]
        
        return render_template('index.html', prediction_text=f'Predicted House Price: ${predicted_price:,.2f}')
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="Error: Unable to make prediction. Check server logs for details.")

if __name__ == "__main__":
    app.run(debug=True)
