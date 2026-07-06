import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

#importo le features e i label utilizzati per l'addestramento
features = np.load("FEATURES.NPY") 
Y = np.load("Y.npy")

#importo il modello addestrato
model = joblib.load("rf_model.joblib")

#ricreo lo stesso split utilizzato nel training.
X_train, X_test, y_train, y_test = train_test_split(
    features, Y, test_size=0.3, random_state=42, stratify=Y
)

y_pred = model.predict(X_test)
report = classification_report(y_test,y_pred,output_dict=True, zero_division = 0)

#stampo tutte le metriche.
print("Accuracy:", report["accuracy"])
print("Precision:", report["weighted avg"]["precision"])
print("Recall:", report["weighted avg"]["recall"])
print("F1:", report["weighted avg"]["f1-score"])

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
