def detect_threats(model, X_test):

    predictions = model.predict(X_test)

    print("\n🚨 Threat Detection Results:\n")

    for i, pred in enumerate(predictions[:30]):
        if pred == 1:
            print(f"[ALERT] Attack detected at row {i}")
        else:
            print(f"[SAFE] Normal traffic at row {i}")