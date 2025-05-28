import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset 
df=pd.read_csv("C:/Users/lenovo/OneDrive/Desktop/house price data.csv")
# Clean data: drop nulls, duplicates, and zero/negative prices
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[df['price'] > 0]

# Drop irrelevant columns
df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1, inplace=True)

# Encode categorical features (if these are categorical, otherwise skip or adjust)
cat_cols = ['view', 'waterfront', 'condition']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Separate features and target
X = df.drop('price', axis=1)
y = np.log1p(df['price'])  # Log transform target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict on train and test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate performance
print("Train R2:", r2_score(y_train, y_train_pred))
print("Test R2:", r2_score(y_test, y_test_pred))
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))

# Inverse transform predictions for interpretation
y_test_exp = np.expm1(y_test)
y_test_pred_exp = np.expm1(y_test_pred)
comparison = pd.DataFrame({
    'Actual Price': np.floor(y_test_exp),
    'Predicted Price': np.floor(y_test_pred_exp)
})
comparison.reset_index(drop=True, inplace=True)
print(comparison.head())
