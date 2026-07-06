import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# carico feature e label
features = np.load("FEATURES.npy")
Y = np.load("Y.npy")

# carico modello e scaler salvati precedentemente
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")

# ricreo lo stesso split usato nel training
X_train, X_test, y_train, y_test = train_test_split(
    features, Y, test_size=0.3, random_state=42, stratify=Y
)

# applico lo stesso scaling del training
X_test_scaled = scaler.transform(X_test)

# predizione
y_pred = model.predict(X_test_scaled)

# metriche
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

#stampo tutte le metriche.
print("Accuracy:", report["accuracy"])
print("Precision:", report["weighted avg"]["precision"])
print("Recall:", report["weighted avg"]["recall"])
print("F1:", report["weighted avg"]["f1-score"])

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))