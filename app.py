from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("Housing.csv")

# Define features and target
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
            'basement', 'hotwaterheating', 'airconditioning', 'parking',
            'prefarea', 'furnishingstatus']
X = df[features]
y = df['price']

# Preprocessing
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                        'airconditioning', 'prefarea', 'furnishingstatus']
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), categorical_features)],
    remainder='passthrough'  # Keep numerical features
)

# Preprocess and train the model
X_processed = preprocessor.fit_transform(X)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_processed, y)

# Get list of possible values for dropdowns
dropdown_options = {
    'mainroad': sorted(df['mainroad'].unique()),
    'guestroom': sorted(df['guestroom'].unique()),
    'basement': sorted(df['basement'].unique()),
    'hotwaterheating': sorted(df['hotwaterheating'].unique()),
    'airconditioning': sorted(df['airconditioning'].unique()),
    'prefarea': sorted(df['prefarea'].unique()),
    'furnishingstatus': sorted(df['furnishingstatus'].unique())
}

@app.route('/')
def home():
    return render_template('index.html', options=dropdown_options)

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    input_data = {
        'area': int(request.form['area']),
        'bedrooms': int(request.form['bedrooms']),
        'bathrooms': int(request.form['bathrooms']),
        'stories': int(request.form['stories']),
        'mainroad': request.form['mainroad'],
        'guestroom': request.form['guestroom'],
        'basement': request.form['basement'],
        'hotwaterheating': request.form['hotwaterheating'],
        'airconditioning': request.form['airconditioning'],
        'parking': int(request.form['parking']),
        'prefarea': request.form['prefarea'],
        'furnishingstatus': request.form['furnishingstatus']
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Transform input
    input_processed = preprocessor.transform(input_df)

    # Predict
    predicted_price = model.predict(input_processed)[0]
    predicted_price = round(predicted_price, 2)

    return render_template('index.html', prediction_text=f"Predicted House Price: ₹ {predicted_price}", options=dropdown_options)

if __name__ == '__main__':
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")

    # For local development (you can remove this in production if not needed)
    threading.Timer(1.25, open_browser).start()

    # Use dynamic port from environment or default to 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
