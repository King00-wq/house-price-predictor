from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)

# Load Random Forest model
with open('model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load Linear Regression model + scaler
with open('lr_model.pkl', 'rb') as f:
    lr_bundle = pickle.load(f)
    lr_model  = lr_bundle['model']
    scaler    = lr_bundle['scaler']

# Load metrics
with open('metrics.json', 'r') as f:
    metrics = json.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        overall_qual = float(data.get('OverallQual', 5))
        gr_liv_area  = float(data.get('GrLivArea', 1500))
        total_sf     = float(data.get('TotalSF', 2000))
        garage_cars  = float(data.get('GarageCars', 2))
        total_bsmt   = float(data.get('TotalBsmtSF', 800))
        year_built   = float(data.get('YearBuilt', 2000))
        total_bath   = float(data.get('TotalBath', 2))
        lot_area     = float(data.get('LotArea', 8000))
        house_age    = 2024 - year_built
        remod_age    = house_age

        features = [[overall_qual, gr_liv_area, total_sf, garage_cars,
                     total_bsmt, house_age, total_bath, lot_area,
                     year_built, remod_age]]

        rf_pred = rf_model.predict(features)[0]
        lr_pred = lr_model.predict(scaler.transform(features))[0]

        return jsonify({
            'rf_price': round(float(rf_pred), 0),
            'lr_price': round(float(lr_pred), 0),
            'price': round(float(rf_pred), 0),   # default for legacy
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})


@app.route('/teacher')
def teacher():
    return render_template('teacher.html', metrics=metrics)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
