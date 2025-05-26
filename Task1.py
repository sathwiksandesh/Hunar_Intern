#importing pandas package
import pandas as pd
# loading the data set
df=pd.read_csv("C:/Users/lenovo/OneDrive/Desktop/food_coded.csv")
df.info()
# Remove duplicate rows
df = df.drop_duplicates()
# Remove duplicate columns
df = df.loc[:, ~df.T.duplicated()]
# Fill missing values with the mean -numeric columns only
df = df.fillna(df.mean(numeric_only=True))
# Fill missing values with the median -numeric columns only
df = df.fillna(df.median(numeric_only=True))
# Fill missing values with the mode -column by column
for col in df.columns:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val[0])
df.info()
