import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import urllib.request

TRAIN_URL = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
TEST_URL = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"

def download_data(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
    
    if not os.path.exists(train_path):
        print(f"Downloading {train_path}...")
        try:
            urllib.request.urlretrieve(TRAIN_URL, train_path)
        except Exception as e:
            print(f"Failed to download from {TRAIN_URL}: {e}")
            print("Please download UNSW_NB15_training-set.csv manually and place it in the 'data' folder.")
            
    if not os.path.exists(test_path):
        print(f"Downloading {test_path}...")
        try:
            urllib.request.urlretrieve(TEST_URL, test_path)
        except Exception as e:
            print(f"Failed to download from {TEST_URL}: {e}")
            print("Please download UNSW_NB15_testing-set.csv manually and place it in the 'data' folder.")

    return train_path, test_path

def load_and_preprocess(train_path, test_path):
    print("Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Drop IDs if they exist
    if 'id' in df_train.columns:
        df_train = df_train.drop(columns=['id'])
    if 'id' in df_test.columns:
        df_test = df_test.drop(columns=['id'])
        
    # The dataset has 'label' (0 or 1) and 'attack_cat' (category)
    # We drop attack_cat because our task is binary classification.
    if 'attack_cat' in df_train.columns:
        df_train = df_train.drop(columns=['attack_cat'])
    if 'attack_cat' in df_test.columns:
        df_test = df_test.drop(columns=['attack_cat'])
        
    print(f"Train set shape after dropping IDs/attack_cat: {df_train.shape}")
    
    # Combine for consistent encoding
    df_all = pd.concat([df_train, df_test], axis=0)
    
    # Identify categorical columns
    categorical_cols = df_all.select_dtypes(include=['object']).columns
    print(f"Categorical columns to encode: {categorical_cols.tolist()}")
    
    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str))
        
    # Split back
    df_train_encoded = df_all.iloc[:len(df_train)].copy()
    df_test_encoded = df_all.iloc[len(df_train):].copy()
    
    # Separate features and target
    X_train_full = df_train_encoded.drop(columns=['label'])
    y_train_full = df_train_encoded['label']
    
    X_test = df_test_encoded.drop(columns=['label'])
    y_test = df_test_encoded['label']
    
    # Standard scale features
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep feature names
    X_train_full_scaled = pd.DataFrame(X_train_full_scaled, columns=X_train_full.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Subsample for faster execution
    SAMPLE_SIZE = 10000
    if len(X_train_full_scaled) > SAMPLE_SIZE:
        X_train_full_scaled, _, y_train_full, _ = train_test_split(
            X_train_full_scaled, y_train_full, train_size=SAMPLE_SIZE, random_state=42, stratify=y_train_full
        )
    if len(X_test_scaled) > SAMPLE_SIZE:
        X_test_scaled, _, y_test, _ = train_test_split(
            X_test_scaled, y_test, train_size=SAMPLE_SIZE, random_state=42, stratify=y_test
        )
    
    # Split development partition into fit subset and selection-validation subset
    # Let's say 80% fit, 20% validation
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full_scaled, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    return {
        'X_fit': X_fit, 'y_fit': y_fit,
        'X_val': X_val, 'y_val': y_val,
        'X_train_full': X_train_full_scaled, 'y_train_full': y_train_full,
        'X_test': X_test_scaled, 'y_test': y_test,
        'feature_names': X_train_full.columns.tolist()
    }
