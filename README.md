# 🏠 House Price Prediction Web App

This repository contains a machine learning project to predict house prices based on various features. The prediction model is deployed using Flask and trained on the **Housing.csv** dataset using a **Random Forest Regressor** (preferred model).



## 📁 Contents

- `preprocessing.py` – Performs data cleaning, transformation, feature engineering, and scaling.
- `app.py` – **Flask web app using Random Forest Regressor** (recommended version).
- `linearregression.py` – Optional: Flask web app using Linear Regression for comparison.
- `Housing.csv` – Dataset used for training the models.
- `templates/index.html` – HTML template for `app.py`.
- `templates/index2.html` – HTML template for `linearregression.py`.

---

## 📊 Dataset

The dataset `Housing.csv` includes features such as:

- Area (in sq.ft)
- Bedrooms
- Bathrooms
- Stories
- Main road access
- Guest room availability
- Basement availability
- Hot water heating
- Air conditioning
- Parking space
- Preferred area
- Furnishing status
- Price (Target variable)

---

## 🧹 Data Preprocessing

- **Label Encoding**: For binary categorical features (e.g., yes/no).
- **One-Hot Encoding**: For features like `furnishingstatus`.
- **Feature Scaling**: StandardScaler applied to numerical features.
- **Feature Engineering**:
  - `price_per_sqft` = price / area
  - `total_rooms` = bedrooms + bathrooms



## 🤖 Models

### ✅ Preferred: Random Forest Regressor
- Implemented in `app.py`
- Ensemble-based model using 100 trees for robust performance

### Optional: Linear Regression
- Implemented in `linearregression.py`
- Simpler, useful for comparison and interpretability

---

## 🌐 Web App Features

- Built using **Flask**
- Users can input house features via a form
- The app returns the **predicted price** instantly

---

## ▶️ How to Run the App

Install the required packages:


pip install flask pandas scikit-learn matplotlib seaborn
Recommended: Run Random Forest App

python app.py
Visit the app at [http://127.0.0.1:5000](http://127.0.0.1:5000)

Example Output:
Predicted House Price: ₹ 1,234,567.89
📜 License
This project is licensed under the MIT License.

Author:
P.Subha Paramesh
B.E. CSE Student
