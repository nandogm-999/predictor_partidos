import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

DATA_PATH = "data/processed_data_last_3.csv"
FEATURES = [
    "diff_elo",
    "diff_avg_goals_for",
    "diff_avg_goals_against",
    "diff_avg_points",
    "diff_ranking",
    "diff_fifa_points",
    "ranking_local",
    "ranking_away",
    "change_local",
    "change_away",
]


def main():
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=FEATURES)

    # Split temporal (validacion con datos de 2024 a junio 2026)
    df_train = df[df["date"] < "2024-01-01"]
    df_test = df[df["date"] >= "2024-01-01"]

    X_train = df_train[FEATURES]
    y_train = df_train["result"]
    X_test = df_test[FEATURES]
    y_test = df_test["result"]

    # Modelo temporal para sacar metricas
    eval_model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=70,
        random_state=45,
        class_weight='balanced'
    )
    eval_model.fit(X_train, y_train)

    y_pred = eval_model.predict(X_test)
    y_score = eval_model.predict_proba(X_test)

    # Metricas globales y detalladas en consola
    acc = accuracy_score(y_test, y_pred)
    print("Model Performance Metrics (Test Set: 2024-01-01 to 2026-06-01):")
    print(f"Accuracy: {acc:.2%}\n")
    print(classification_report(y_test, y_pred))

    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    classes = ["Draw (0)", "Home Win (1)", "Away Win (2)"]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (2024-2026)")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # Curvas ROC multiclase
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8, 6))
    labels_roc = ["Draw (0)", "Home Win (1)", "Away Win (2)"]
    colors = ["orange", "green", "blue"]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=2,
            label=f"ROC curve of {labels_roc[i]} (AUC = {roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Multiclass ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/roc_curve.png")
    plt.close()



if __name__ == "__main__":
    main()