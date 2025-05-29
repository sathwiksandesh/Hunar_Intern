# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
#Loading the dataset
df = pd.read_csv("C:/Users/lenovo/OneDrive/Desktop/breast cancer.csv") 
print(df.head())
print(df.info())
# Separating features and target
# Assuming 'diagnosis' column: 'M' = malignant, 'B' = benign
X = df.drop(columns=['diagnosis', 'id', 'Unnamed: 32'], errors='ignore')  # Drop non-feature columns
y = df['diagnosis'].map({'M': 0, 'B': 1})  # Encode target: M=0, B=1
# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
# 5. Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
# 6. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train the KNN model
k = 7
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
# Evaluate the model
y_pred = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
# Elbow method for best k
error_rates = []
k_values = range(1, 21)

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    pred_k = model.predict(X_test_scaled)
    error = 1 - accuracy_score(y_test, pred_k)
    error_rates.append(error)

plt.figure(figsize=(8, 4))
plt.plot(k_values, error_rates, marker='o', linestyle='-', color='blue')
plt.title("📐 Elbow Method for Optimal k")
plt.xlabel("k")
plt.ylabel("Error Rate")
plt.xticks(k_values)
plt.grid(True)
plt.show()
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Malignant', 'Benign'], cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Try k values from 1 to 20
k_values = list(range(1, 21))
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred_k = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='navy')
plt.title("📐 Accuracy vs. k (Number of Neighbors)")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.axvline(x=7, color='red', linestyle='--', label='k = 7')
plt.legend()
plt.tight_layout()
plt.show()
# Print the best k
best_k = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f" Best k = {best_k} with Accuracy = {best_acc:.4f}")

