import pickle
import os
import pandas as pd
from MyTransformers import PreProcessingTransformer
from flask import Flask, request

# Load Model
model=pickle.load(open('models/modelo_tunned_random.pkl', 'rb'))

# Instanciando o Flask
app=Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    pp = PreProcessingTransformer()
    test_json = request.get_json()
    
    # Collect data
    if test_json:
        if isinstance(test_json, dict):
            df_raw = pd.DataFrame(test_json, index=[0])
            df_raw_processed = pp.fit_transform(df_raw)
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            df_raw_processed = pp.fit_transform(df_raw)
            
    # Predictions
    pred = model.predict(df_raw_processed)
    df_raw_processed['prediction'] = pred
    return df_raw_processed.to_json(orient='records')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)