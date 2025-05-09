import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#ds
df = pd.read_csv("Housing.csv")

# Label encode
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishingstatus
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)


numerical_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

#Histogram
df[numerical_cols].hist(bins=10, figsize=(12, 8))
plt.suptitle("Histogram of Numerical Features", fontsize=16)
plt.show()

#Box Plot
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df[col])
    plt.title(col)
plt.tight_layout()
plt.suptitle("Boxplots for Outlier Detection", fontsize=16, y=1.02)
plt.show()

#Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


