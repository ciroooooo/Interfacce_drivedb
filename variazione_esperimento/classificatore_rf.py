import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

FEATURES = np.load("FEATURES.npy")
Y = np.load("Y.npy")

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    FEATURES, Y, test_size=0.3, random_state=42, stratify=Y
)

# Random Forest non richiede scaling obbligatorio
rf = RandomForestClassifier(random_state=1)

# griglia piccola ma sensata, includendo i parametri del paper
param_grid = {
    "n_estimators": [100, 110, 150],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# prova tutte le combinazioni possibili con cross-validation a 5 fold: divido in 5 parti, 4 allenamento 1 test.
# sceglie la combinazione con l'accuracy media piu alta
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)

# test
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Stampa dei risultati
print("Best params:")
print(grid.best_params_)
print("\nTest accuracy:")
print(grid.score(X_test, y_test))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))


# salvataggio
joblib.dump(best_model, "rf_model.joblib")