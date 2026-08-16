"""
Model training script for ASL Sign Language Gesture Recognition.

This script loads the pre-processed landmark features, splits them into
training, validation, and testing sets, and trains multiple classifiers
including Random Forest, SVM, scikit-learn MLP, and an optional Keras MLP.
It selects the best performing model based on validation accuracy and evaluates
it on the test set, saving the model, encoder, scaler, and evaluation artifacts.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found. Keras MLP model will not be trained.")

# Import configuration
try:
    from config import (
        DATA_FILE, MODELS_DIR, ALL_LABELS, TOTAL_FEATURES,
        TEST_SIZE, VAL_SIZE, RANDOM_STATE,
        MLP_HIDDEN_LAYERS, MLP_MAX_ITER, MLP_EPOCHS, MLP_LEARNING_RATE, MLP_BATCH_SIZE,
        BEST_MODEL_PATH, MLP_MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH,
        TRAINING_REPORT_PATH, CONFUSION_MATRIX_PATH
    )
    from utils import FeatureExtractor
except ImportError as e:
    print(f"Error importing config or utils: {e}")
    sys.exit(1)


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, StandardScaler, pd.DataFrame]:
    """Loads CSV, explores data, encodes labels, scales features, and splits."""
    print("=" * 50)
    print("Loading and Preprocessing Data")
    print("=" * 50)
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        sys.exit(1)
        
    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = [f"feature_{i}" for i in range(TOTAL_FEATURES)] + ["label"]
    
    # Feature columns
    feature_names = FeatureExtractor().get_feature_names()
    if len(feature_names) != TOTAL_FEATURES:
        print(f"Warning: Expected {TOTAL_FEATURES} features, but got {len(feature_names)}")
        
    # Check if we have enough data
    if len(df) == 0:
        print("Error: Dataset is empty.")
        sys.exit(1)
        
    if 'label' not in df.columns:
        print("Error: 'label' column not found in dataset.")
        sys.exit(1)
        
    # Dataset statistics
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {df['label'].nunique()}")
    print("\nClass distribution:")
    class_counts = df['label'].value_counts()
    print(class_counts)
    
    if class_counts.min() < 10:
        print("\nWarning: Some classes have very few samples. This may cause issues during stratified splitting.")
        
    # Separate features and labels
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split: Train vs (Val + Test)
    val_test_size = VAL_SIZE + TEST_SIZE
    
    # Handle case where stratify might fail due to small class counts
    stratify = y_encoded if class_counts.min() > 1 else None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=val_test_size, random_state=RANDOM_STATE, stratify=stratify
    )
    
    # Split: Val vs Test
    relative_test_size = TEST_SIZE / val_test_size
    
    stratify_temp = y_temp if (stratify is not None and pd.Series(y_temp).value_counts().min() > 1) else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=RANDOM_STATE, stratify=stratify_temp
    )
    
    print("\nSplit statistics:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, encoder, scaler, df

def train_keras_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, num_classes: int) -> Any:
    """Trains a Keras sequential model."""
    if not TF_AVAILABLE:
        return None
        
    print("\nTraining Keras MLP Model...")
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Using config learning rate if defined, otherwise default Adam
    opt = tf.keras.optimizers.Adam(learning_rate=MLP_LEARNING_RATE)
    
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
                  
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0
    )
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Keras MLP Validation Accuracy: {val_acc:.4f}")
    
    return model


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, encoder, scaler, df = load_and_preprocess_data()
    
    num_classes = len(encoder.classes_)
    target_names = encoder.classes_
    
    # Models to train
    models: Dict[str, Any] = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=30, random_state=RANDOM_STATE, n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_STATE),
        'Scikit-learn MLP': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=MLP_MAX_ITER, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15)
    }
    
    print("\n" + "=" * 50)
    print("Training Models")
    print("=" * 50)
    
    results = {}
    reports = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        print(f"{name} Validation Accuracy: {val_acc:.4f}")
        
        results[name] = val_acc
        report = classification_report(y_val, y_val_pred, target_names=target_names, zero_division=0)
        reports[name] = report
        
    # Train Keras Model
    keras_model = train_keras_mlp(X_train, y_train, X_val, y_val, num_classes)
    if keras_model is not None:
        y_val_pred_prob = keras_model.predict(X_val, verbose=0)
        y_val_pred = np.argmax(y_val_pred_prob, axis=1)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        name = 'Keras MLP'
        results[name] = val_acc
        report = classification_report(y_val, y_val_pred, target_names=target_names, zero_division=0)
        reports[name] = report
        models[name] = keras_model
        
    # Select Best Model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_val_acc = results[best_model_name]
    
    print("\n" + "=" * 50)
    print("Model Evaluation Summary (Validation Set)")
    print("=" * 50)
    
    print(f"{'Model Name':<25} | {'Validation Accuracy':<20}")
    print("-" * 50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<25} | {acc:.4f}")
        
    print(f"\nSelected Best Model: {best_model_name} with Validation Accuracy: {best_val_acc:.4f}")
    
    print("\n" + "=" * 50)
    print("Testing Best Model")
    print("=" * 50)
    
    # Evaluate Best Model on Test Set
    if best_model_name == 'Keras MLP':
        y_test_pred_prob = best_model.predict(X_test, verbose=0)
        y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    else:
        y_test_pred = best_model.predict(X_test)
        
    test_acc = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred, target_names=target_names, zero_division=0)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nTest Classification Report:")
    print(test_report)
    
    # Create Models Dir if not exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save artifacts
    print("\nSaving Artifacts...")
    joblib.dump(encoder, LABEL_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"- LabelEncoder saved to {LABEL_ENCODER_PATH}")
    print(f"- StandardScaler saved to {SCALER_PATH}")
    
    if best_model_name == 'Keras MLP':
        best_model.save(MLP_MODEL_PATH)
        print(f"- Best Keras Model saved to {MLP_MODEL_PATH}")
    else:
        joblib.dump(best_model, BEST_MODEL_PATH)
        print(f"- Best {best_model_name} Model saved to {BEST_MODEL_PATH}")
        
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {best_model_name} (Test Set)')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"- Confusion Matrix saved to {CONFUSION_MATRIX_PATH}")
    
    # Save Training Report
    with open(TRAINING_REPORT_PATH, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Validation samples: {X_val.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n\n")
        
        f.write("-" * 50 + "\n")
        f.write("Validation Results\n")
        f.write("-" * 50 + "\n")
        for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{name}: {acc:.4f}\n")
            f.write(reports[name] + "\n")
            
        f.write("\n" + "-" * 50 + "\n")
        f.write(f"Test Results for Best Model ({best_model_name})\n")
        f.write("-" * 50 + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(test_report)
        
    print(f"- Training Report saved to {TRAINING_REPORT_PATH}")
    
    print("\nTraining Pipeline Completed Successfully.")

if __name__ == '__main__':
    main()
