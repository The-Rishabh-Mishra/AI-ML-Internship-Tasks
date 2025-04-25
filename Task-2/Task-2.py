import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Load your dataset
import pandas as pd

# Adjust path as needed
df = pd.read_csv(r"C:\Users\thema\OneDrive\Documents\Desktop\AI-ML Internships\Task-2\Titanic-Dataset.csv")

print(df.head())  # Show first few rows
  # Replace with your filename

# 1. Summary statistics
print("Summary Statistics:\n", df.describe())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# 2. Histograms & Boxplots for numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Histograms
df[numeric_cols].hist(bins=30, figsize=(12, 10))
plt.suptitle('Histograms of Numeric Features')
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
df[numeric_cols].plot(kind='box', subplots=True, layout=(1, len(numeric_cols)), figsize=(15, 5), sharey=False)
plt.suptitle('Boxplots of Numeric Features')
plt.tight_layout()
plt.show()

# 3. Correlation Matrix & Pairplot
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pairplot (might be heavy for large datasets)
# sns.pairplot(df[numeric_cols])
# plt.show()

# 4. Detecting patterns, trends, or anomalies
# Example: Time trend if time column exists
df['date'] = pd.to_datetime(df['date_column'])  # Uncomment & adjust
df.set_index('date')['some_numeric_col'].plot()
plt.title('Trend Over Time')
plt.show()

# 5. Inferences (printed as observations)
print("\nPotential Inferences:")
print("- Look at the correlation matrix for linear relationships.")
print("- Histograms show skewed vs normal distributions.")
print("- Boxplots can help detect outliers.")


fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="Test Plot")
fig.show()