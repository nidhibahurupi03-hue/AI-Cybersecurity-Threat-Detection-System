from src1.data_preprocessing import load_data, preprocess_data
from src1.model import train_model
from src1.evaluate import evaluate_model

def main():
    print("\n🚀 ===============================")
    print("   AI CYBERSECURITY SYSTEM STARTED")
    print("================================\n")

    # Load dataset (LOCAL FILE ONLY - NO UPLOAD ISSUE)
    df = load_data("data/network_data.csv")
    print("✅ Dataset Loaded Successfully")

    # 🔥 FIX: prevent large dataset crash
    df = df.sample(min(5000, len(df)))

    print(f"📊 Dataset reduced to: {len(df)} rows")

    # Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("✅ Data Preprocessing Completed")

    # Model training
    model = train_model(X_train, y_train)
    print("✅ Model Training Completed")

    # Evaluation + Visualization
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()