# titanic_preprocessing.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# STEP 1: Basic Info
print("Initial Info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# STEP 2: Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Too many missing values

# STEP 3: Encode Categorical Features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Drop non-essential columns
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# STEP 4: Normalize Numerical Features
scaler = StandardScaler()
num_cols = ['Age', 'Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])

# STEP 5: Visualize and Remove Outliers (Boxplot + IQR method)
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Final Check
print("\nData after cleaning and preprocessing:")
print(df.head())
print("\nFinal shape:", df.shape)
