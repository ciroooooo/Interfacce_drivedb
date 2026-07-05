import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FEATURES = np.load("FEATURES.npy")
Y = np.load("Y.npy")

# stesso split del classificatore
X_train, X_test, y_train, y_test = train_test_split(
    FEATURES, Y, test_size=0.3, random_state=42, stratify=Y
)

# usa i best params trovati nel classificatore_rf.py
model = RandomForestClassifier(
    n_estimators=110,
    criterion="entropy",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=1
)

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

    accs.append(report["accuracy"])
    precisions.append(report["weighted avg"]["precision"])
    recalls.append(report["weighted avg"]["recall"])
    f1s.append(report["weighted avg"]["f1-score"])

    print(f"\nFold {fold}")
    print(cm)
    print(classification_report(y_val, y_pred, zero_division=0))

# =========================
# Heatmap confusion matrix per fold
# =========================
fig_cm = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=[
        f"Fold 1<br>Acc {accs[0]:.2f}",
        f"Fold 2<br>Acc {accs[1]:.2f}",
        f"Fold 3<br>Acc {accs[2]:.2f}",
        f"Fold 4<br>Acc {accs[3]:.2f}",
        f"Fold 5<br>Acc {accs[4]:.2f}",
        ""
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

fig_cm.update_xaxes(visible=False, row=2, col=3)
fig_cm.update_yaxes(visible=False, row=2, col=3)

for r in [1, 2]:
    for c in [1, 2, 3]:
        fig_cm.update_xaxes(title_text="Pred", row=r, col=c)
        fig_cm.update_yaxes(title_text="True", row=r, col=c)

fig_cm.update_layout(
    title="Random Forest confusion matrix per fold",
    font=dict(size=15),
    height=800,
    width=1100
)

fig_cm.show()

# se vuoi salvarla
# fig_cm.write_image("rf_confusion_folds.png")

# =========================
# Grafico metriche per fold
# =========================
folds = [1, 2, 3, 4, 5]

fig_metrics = go.Figure()

fig_metrics.add_trace(go.Scatter(x=folds, y=accs, mode="lines+markers", name="Accuracy"))
fig_metrics.add_trace(go.Scatter(x=folds, y=precisions, mode="lines+markers", name="Precision"))
fig_metrics.add_trace(go.Scatter(x=folds, y=recalls, mode="lines+markers", name="Recall"))
fig_metrics.add_trace(go.Scatter(x=folds, y=f1s, mode="lines+markers", name="F1-score"))

fig_metrics.update_layout(
    title="Random Forest metriche per fold",
    xaxis_title="Fold",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    font=dict(size=16)
)

fig_metrics.show()

# se vuoi salvarlo
# fig_metrics.write_image("rf_metrics_folds.png")

print("\n=== Media metriche sui 5 fold ===")
print(f"Accuracy media:  {np.mean(accs):.4f}")
print(f"Precision media: {np.mean(precisions):.4f}")
print(f"Recall media:    {np.mean(recalls):.4f}")
print(f"F1 media:        {np.mean(f1s):.4f}")

# salva in locale
fig_cm.write_image("rf_confusion_folds.png")
fig_metrics.write_image("rf_metrics_folds.png")

print("Immagini salvate:")
print("- rf_confusion_folds.png")
print("- rf_metrics_folds.png")