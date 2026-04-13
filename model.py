from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "model1/model.pkl")

    return model