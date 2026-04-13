import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):

    # 1. clean column names
    df.columns = df.columns.str.strip()

    # 2. remove missing values
    df = df.dropna()

    # 3. replace infinity values (IMPORTANT FIX)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 4. find label column automatically
    label_col = None
    for col in df.columns:
        if col.lower() in ["label", "class", "target"]:
            label_col = col
            break

    if label_col is None:
        raise Exception("Label column not found in dataset")

    # 5. encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if col != label_col:
            df[col] = LabelEncoder().fit_transform(df[col])

    # 6. split features and target
    X = df.drop(label_col, axis=1)
    y = df[label_col]

    # 7. final safety cleanup
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # 8. train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)