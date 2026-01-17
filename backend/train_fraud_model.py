"""
Fraud Detection Model Training Script
Trains neural network and displays comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def load_data():
    """Load synthetic fraud data"""
    filepath = 'data/training_data/fraud_transactions.csv'
    
    if not os.path.exists(filepath):
        print(f"❌ Data file not found: {filepath}")
        print("   Run generate_synthetic_data.py first!")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    
    # Separate features and labels
    X = df.drop('is_fraud', axis=1).values
    y = df['is_fraud'].values
    
    return X, y, df

def build_model(input_dim=10):
    """Build neural network architecture"""
    model = Sequential([
        # Input + Hidden Layer 1
        Dense(32, activation='relu', input_shape=(input_dim,), name='hidden_1'),
        Dropout(0.3, name='dropout_1'),
        
        # Hidden Layer 2
        Dense(16, activation='relu', name='hidden_2'),
        Dropout(0.2, name='dropout_2'),
        
        # Hidden Layer 3
        Dense(8, activation='relu', name='hidden_3'),
        
        # Output Layer
        Dense(1, activation='sigmoid', name='output')
    ])
    
    return model

def train_model():
    """Main training function"""
    
    print_header("FRAUD DETECTION MODEL TRAINING")
    
    # Load data
    print("\n📊 Loading synthetic data...")
    X, y, df = load_data()
    
    print(f"✅ Loaded {len(X)} transactions")
    print(f"   - Legitimate: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"   - Fraud: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # Split data
    print("\n📂 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   - Training set: {len(X_train)} samples")
    print(f"   - Test set: {len(X_test)} samples")
    
    # Scale features
    print("\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    print_header("BUILDING NEURAL NETWORK")
    model = build_model(input_dim=X_train.shape[1])
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n🧠 Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'backend/trained_models/fraud_model_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    # Train
    print_header("TRAINING MODEL")
    print("⏳ Training in progress...\n")
    
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print_header("EVALUATION METRICS")
    
    # Predictions
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📊 CONFUSION MATRIX:")
    print(f"""
                        Predicted
                    No Fraud  |  Fraud
        Actual  ────────────────────────────
        No Fraud    {tn:6d}    |  {fp:6d}  ← False Positive (FP)
        Fraud       {fn:6d}    |  {tp:6d}  ← True Positive (TP)
                        ↑
                False Negative (FN)
    """)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Error rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    print("\n📈 PERFORMANCE METRICS:")
    print(f"   Precision:  {precision:.2%}  {'✅' if precision > 0.80 else '❌'} (Target: >80%)")
    print(f"   Recall:     {recall:.2%}  {'✅' if recall > 0.95 else '❌'} (Target: >95%)")
    print(f"   F1-Score:   {f1:.2%}  {'✅' if f1 > 0.85 else '❌'} (Target: >85%)")
    print(f"   AUC-ROC:    {auc:.2%}  {'✅' if auc > 0.90 else '❌'} (Target: >90%)")
    
    print("\n⚠️  ERROR RATES:")
    print(f"   False Positive Rate (FPR): {fpr:.2%}  {'✅' if fpr < 0.10 else '❌'} (Target: <10%)")
    print(f"   False Negative Rate (FNR): {fnr:.2%}  {'✅' if fnr < 0.05 else '❌'} (Target: <5%)")
    
    # Detailed definitions
    print_header("METRIC DEFINITIONS")
    print("""
    TRUE POSITIVE (TP) = {tp}
      Model predicted: Fraud | Actually was: Fraud
      ✅ GOOD! We caught real fraud
    
    TRUE NEGATIVE (TN) = {tn}
      Model predicted: Legitimate | Actually was: Legitimate
      ✅ GOOD! We didn't bother user
    
    FALSE POSITIVE (FP) = {fp}
      Model predicted: Fraud | Actually was: Legitimate
      ⚠️  BAD! User annoyed by false alarm
      
    FALSE NEGATIVE (FN) = {fn}
      Model predicted: Legitimate | Actually was: Fraud
      🚨 VERY BAD! We missed fraud, user loses money!
    
    PRECISION = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.2%}
      "Of all fraud alerts, how many were real?"
      
    RECALL = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.2%}
      "Of all actual fraud, how many did we catch?"
      
    F1-SCORE = 2 * (Precision * Recall) / (Precision + Recall) = {f1:.2%}
      "Harmonic mean - balanced measure"
    """.format(tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1))
    
    # Real-world interpretation
    print_header("REAL-WORLD INTERPRETATION")
    print(f"""
    💡 What this means in practice:
    
    Out of 100 ACTUAL fraud transactions:
      - We CATCH: {int(recall*100)} frauds ✅
      - We MISS: {int(fnr*100)} frauds 🚨
    
    Out of 100 fraud ALERTS we send:
      - {int(precision*100)} are REAL fraud ✅
      - {int((1-precision)*100)} are FALSE alarms ⚠️
    
    Out of 100 LEGITIMATE transactions:
      - We correctly identify: {int((1-fpr)*100)} ✅
      - We wrongly flag: {int(fpr*100)} as fraud ⚠️
    
    COST ANALYSIS:
      - Missing 1 fraud (FN): User loses ~₹50,000 💸
      - 1 false alarm (FP): User annoyed, no money lost
      - Total FN cost: {fn} × ₹50,000 = ₹{fn * 50000:,}
      - Total FP cost: {fp} × ₹0 = ₹0
      - TOTAL POTENTIAL LOSS: ₹{fn * 50000:,}
    """)
    
    # Classification report
    print_header("DETAILED CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, 
                                target_names=['Legitimate', 'Fraud'],
                                digits=4))
    
    # Save model
    model.save('backend/trained_models/fraud_model.h5')
    print("\n💾 Model saved to: backend/trained_models/fraud_model.h5")
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'backend/trained_models/scaler.pkl')
    print("💾 Scaler saved to: backend/trained_models/scaler.pkl")
    
    print_header("TRAINING COMPLETE!")
    
    # Final verdict
    if recall > 0.95 and precision > 0.80 and f1 > 0.85:
        print("\n🎉 SUCCESS! Model meets all target metrics!")
    else:
        print("\n⚠️  Model needs improvement. Consider:")
        if recall < 0.95:
            print("   - Increase recall: Adjust threshold, add more fraud samples")
        if precision < 0.80:
            print("   - Increase precision: Better feature engineering")
        if f1 < 0.85:
            print("   - Balance precision/recall: Tune hyperparameters")
    
    return model, history, scaler

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('backend/trained_models', exist_ok=True)
    os.makedirs('data/training_data', exist_ok=True)
    
    # Train model
    model, history, scaler = train_model()
