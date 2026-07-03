import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# carico le feature e le label
FEATURES = np.load("FEATURES.npy")
Y = np.load("Y.npy")

# stesso split usato nel classificatore_knn.py
X_train, X_test, y_train, y_test = train_test_split(
    FEATURES, Y, test_size=0.3, random_state=42, stratify=Y
)

# standardizzazione sul training set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# uso i best params trovati prima con GridSearchCV
model = KNeighborsClassifier(
    n_neighbors=5,
    leaf_size=20,
    weights="distance"
)

# 5 fold interni sul training set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cms = []
accs = []
precisions = []
recalls = []
f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    cms.append(cm)

    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    acc = report["accuracy"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    accs.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print(f"\nFold {fold}")
    print(cm)
    print(classification_report(y_val, y_pred, zero_division=0))

# =========================
# 1) Heatmap confusion matrix per fold - versione più leggibile
# =========================
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_cm = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=[
        f"Fold 1<br>Acc {accs[0]:.2f}",
        f"Fold 2<br>Acc {accs[1]:.2f}",
        f"Fold 3<br>Acc {accs[2]:.2f}",
        f"Fold 4<br>Acc {accs[3]:.2f}",
        f"Fold 5<br>Acc {accs[4]:.2f}",
        ""   # sesto spazio vuoto
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.18
)

zmax = max(cm.max() for cm in cms)

positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]

for i, cm in enumerate(cms):
    r, c = positions[i]
    fig_cm.add_trace(
        go.Heatmap(
            z=cm,
            x=["0", "1"],
            y=["1", "0"],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18},
            colorscale="Blues",
            zmin=0,
            zmax=zmax,
            showscale=(i == 4),
            hovertemplate="Pred: %{x}<br>True: %{y}<br>Count: %{z}<extra></extra>"
        ),
        row=r,
        col=c
    )

# nasconde il sesto subplot vuoto
fig_cm.update_xaxes(visible=False, row=2, col=3)
fig_cm.update_yaxes(visible=False, row=2, col=3)

# etichette più pulite
for r in [1, 2]:
    for c in [1, 2, 3]:
        fig_cm.update_xaxes(title_text="Pred", row=r, col=c)
        fig_cm.update_yaxes(title_text="True", row=r, col=c)

fig_cm.update_layout(
    title="KNN confusion matrix per fold",
    font=dict(size=15),
    height=800,
    width=1100
)

fig_cm.show()
fig_cm.write_image("knn_confusion_folds_better.png")
# =========================
# 2) Grafico metriche per fold
# =========================
folds = [1, 2, 3, 4, 5]

fig_metrics = go.Figure()

fig_metrics.add_trace(go.Scatter(
    x=folds, y=accs, mode="lines+markers", name="Accuracy"
))
fig_metrics.add_trace(go.Scatter(
    x=folds, y=precisions, mode="lines+markers", name="Precision"
))
fig_metrics.add_trace(go.Scatter(
    x=folds, y=recalls, mode="lines+markers", name="Recall"
))
fig_metrics.add_trace(go.Scatter(
    x=folds, y=f1s, mode="lines+markers", name="F1-score"
))

fig_metrics.update_layout(
    title="KNN metriche per fold (5-fold CV)",
    xaxis_title="Fold",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    font=dict(size=16)
)

fig_metrics.write_image("knn_metrics_folds.png")

# =========================
# riepilogo finale
# =========================
print("\n=== Media metriche sui 5 fold ===")
print(f"Accuracy media:  {np.mean(accs):.4f}")
print(f"Precision media: {np.mean(precisions):.4f}")
print(f"Recall media:    {np.mean(recalls):.4f}")
print(f"F1 media:        {np.mean(f1s):.4f}")