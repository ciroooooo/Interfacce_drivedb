import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
FEATURES = np.load("FEATURES.npy")
Y = np.load("Y.npy")

X_train, X_test, y_train, y_test = train_test_split(FEATURES, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {"n_neighbors": [3, 5, 7], "leaf_size": [20, 30, 40], "weights": ["uniform", "distance"]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_, grid.score(X_test, y_test))
y_pred = grid.predict(X_test)


joblib.dump(grid.best_estimator_, "knn_model.joblib")
joblib.dump(scaler, "scaler.joblib")