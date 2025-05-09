import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Housing.csv")
#Data Cleaning
print(df.isnull().sum())
#there is no missing values 
print("Before Transformation")
print(df)

# Label encoding
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encoding
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Scaling
numerical_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Add new features
df['price_per_sqft'] = df['price'] / df['area']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

print("\nAfter Transformation")
print(df)


