from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # =========================
    # CREATE OUTPUT FOLDER
    # =========================
    os.makedirs("output/graph", exist_ok=True)

    # =========================
    # 1. CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Cybersecurity)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("output/graph/confusion_matrix.png")
    plt.close()

    # =========================
    # 2. NORMALIZED CONFUSION MATRIX
    # =========================
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, cmap="Greens")
    plt.title("Normalized Confusion Matrix")

    plt.savefig("output/graph/confusion_matrix_normalized.png")
    plt.close()

    # =========================
    # 3. ROC CURVE (FIXED)
    # =========================
    if len(np.unique(y_test)) == 2:

        y_prob = model.predict_proba(X_test)[:, 1]

        # convert BENIGN / DDoS → 0 / 1
        if y_test.dtype == 'object':
            y_test_bin = y_test.map({"BENIGN": 0, "DDoS": 1})
        else:
            y_test_bin = y_test

        fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], '--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        plt.savefig("output/graph/roc_curve.png")
        plt.close()

    print("\n✅ Evaluation completed successfully!")