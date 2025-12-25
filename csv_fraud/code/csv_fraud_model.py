import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib

# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================

def load_data(csv_path):
    """Load CSV data and prepare features"""
    # Read the file - it appears to have all columns in one string
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Check if all columns are merged into one
    if len(df.columns) == 1:
        # Split the single column by comma
        first_col_name = df.columns[0]
        
        # Get the proper column names from the header
        col_names = first_col_name.split(',')
        
        # Split each row by comma
        data_rows = []
        for idx, row in df.iterrows():
            row_data = str(row[first_col_name]).split(',')
            data_rows.append(row_data)
        
        # Create new dataframe with split columns
        df = pd.DataFrame(data_rows, columns=col_names)
    
    # Clean column names (remove quotes and whitespace)
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Convert numeric columns to proper types
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['transactions_last_10min'] = pd.to_numeric(df['transactions_last_10min'], errors='coerce')
    df['avg_amount_user'] = pd.to_numeric(df['avg_amount_user'], errors='coerce')
    df['time_since_last_txn_min'] = pd.to_numeric(df['time_since_last_txn_min'], errors='coerce')
    df['amount_deviation_ratio'] = pd.to_numeric(df['amount_deviation_ratio'], errors='coerce')
    df['device_change_flag'] = pd.to_numeric(df['device_change_flag'], errors='coerce')
    df['location_change_flag'] = pd.to_numeric(df['location_change_flag'], errors='coerce')
    
    # Print columns for debugging
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"First few rows:\n{df.head(2)}")
    
    # Parse timestamp to extract hour
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
    df['hour'] = df['timestamp'].dt.hour
    
    return df

# ============================================
# 2. RULE-BASED FRAUD DETECTION
# ============================================

def apply_fraud_rules(df):
    """Apply rule-based fraud detection logic"""
    
    # Initialize rule columns
    df['HIGH_AMOUNT_SPIKE'] = (df['amount_deviation_ratio'] > 5).astype(int)
    df['RAPID_TXN_BURST'] = (df['transactions_last_10min'] > 4).astype(int)
    df['GEO_MISMATCH'] = (df['location_change_flag'] == 1).astype(int)
    df['NEW_DEVICE_HIGH_AMOUNT'] = ((df['device_change_flag'] == 1) & 
                                     (df['amount_deviation_ratio'] > 3)).astype(int)
    df['ODD_HOUR_TXN'] = ((df['hour'].between(1, 5)) & 
                           (df['amount_deviation_ratio'] > 2)).astype(int)
    
    # Additional rules for better detection
    df['LONG_INACTIVITY_SPIKE'] = ((df['time_since_last_txn_min'] > 180) & 
                                     (df['amount_deviation_ratio'] > 4)).astype(int)
    df['VELOCITY_ATTACK'] = ((df['transactions_last_10min'] > 6) & 
                              (df['time_since_last_txn_min'] < 5)).astype(int)
    
    # Count total rules triggered
    rule_columns = ['HIGH_AMOUNT_SPIKE', 'RAPID_TXN_BURST', 'GEO_MISMATCH', 
                    'NEW_DEVICE_HIGH_AMOUNT', 'ODD_HOUR_TXN', 
                    'LONG_INACTIVITY_SPIKE', 'VELOCITY_ATTACK']
    
    df['rules_triggered_count'] = df[rule_columns].sum(axis=1)
    df['rule_triggered'] = df[rule_columns].apply(
        lambda row: ', '.join([col for col, val in row.items() if val == 1]) or 'NONE', 
        axis=1
    )
    
    return df, rule_columns

# ============================================
# 3. TRAIN ISOLATION FOREST MODEL
# ============================================

def train_model(df):
    """Train Isolation Forest for anomaly detection"""
    
    # Select features for ML model
    feature_cols = [
        'amount', 'transactions_last_10min', 'time_since_last_txn_min',
        'amount_deviation_ratio', 'device_change_flag', 'location_change_flag',
        'hour'
    ]
    
    X = df[feature_cols].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    # contamination = expected fraud rate (9%)
    model = IsolationForest(
        contamination=0.09,
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    
    model.fit(X_scaled)
    
    # Get anomaly scores (lower = more anomalous)
    anomaly_scores = model.decision_function(X_scaled)
    
    return model, scaler, anomaly_scores, feature_cols

# ============================================
# 4. FINAL DECISION LOGIC
# ============================================

def make_final_decision(df, anomaly_scores):
    """Combine ML scores with rules for final decision"""
    
    df['anomaly_score'] = anomaly_scores
    
    # Thresholds
    THRESHOLD_LOW = -0.6   # High risk threshold
    THRESHOLD_MID = -0.4   # Medium risk threshold
    
    # Initialize decision columns
    df['risk_level'] = 'LOW'
    df['final_decision'] = 'APPROVED'
    
    # HIGH RISK: Low anomaly score AND 2+ rules triggered
    high_risk_mask = (
        (df['anomaly_score'] < THRESHOLD_LOW) & 
        (df['rules_triggered_count'] >= 2)
    )
    df.loc[high_risk_mask, 'risk_level'] = 'HIGH'
    df.loc[high_risk_mask, 'final_decision'] = 'FLAGGED'
    
    # MEDIUM RISK: Medium anomaly score OR 1+ strong rules triggered
    medium_risk_mask = (
        ((df['anomaly_score'] < THRESHOLD_MID) & (df['anomaly_score'] >= THRESHOLD_LOW)) |
        ((df['rules_triggered_count'] >= 1) & ~high_risk_mask)
    )
    df.loc[medium_risk_mask, 'risk_level'] = 'MEDIUM'
    df.loc[medium_risk_mask, 'final_decision'] = 'FLAGGED'
    
    # Additional strong fraud indicators (override to HIGH)
    strong_fraud_mask = (
        (df['RAPID_TXN_BURST'] == 1) | 
        (df['VELOCITY_ATTACK'] == 1) |
        ((df['HIGH_AMOUNT_SPIKE'] == 1) & (df['GEO_MISMATCH'] == 1))
    )
    df.loc[strong_fraud_mask, 'risk_level'] = 'HIGH'
    df.loc[strong_fraud_mask, 'final_decision'] = 'FLAGGED'
    
    return df

# ============================================
# 5. MAIN TRAINING PIPELINE
# ============================================

def main_training(csv_path, output_model_path='fraud_model.pkl'):
    """Complete training pipeline"""
    
    print("🔄 Loading data...")
    df = load_data(csv_path)
    print(f"✅ Loaded {len(df)} transactions")
    
    print("\n🔄 Applying fraud detection rules...")
    df, rule_columns = apply_fraud_rules(df)
    
    print("\n🔄 Training Isolation Forest model...")
    model, scaler, anomaly_scores, feature_cols = train_model(df)
    
    print("\n🔄 Making final decisions...")
    df = make_final_decision(df, anomaly_scores)
    
    # Save model and scaler
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'rule_columns': rule_columns
    }, output_model_path)
    print(f"\n💾 Model saved to: {output_model_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 TRAINING RESULTS")
    print("="*60)
    
    print(f"\nTotal Transactions: {len(df)}")
    print(f"\nRisk Level Distribution:")
    print(df['risk_level'].value_counts())
    
    print(f"\nFinal Decision Distribution:")
    print(df['final_decision'].value_counts())
    
    print(f"\nFraud Detection Rate: {(df['final_decision'] == 'FLAGGED').sum() / len(df) * 100:.2f}%")
    
    print(f"\nTop Rules Triggered:")
    for rule in rule_columns:
        count = df[rule].sum()
        print(f"  {rule}: {count} ({count/len(df)*100:.1f}%)")
    
    # Save results with predictions
    output_csv = 'fraud_detection_results.csv'
    output_cols = ['transaction_id', 'user_id', 'amount', 'merchant_id', 
                   'actual_location', 'timestamp', 'device_type',
                   'transactions_last_10min', 'time_since_last_txn_min',
                   'amount_deviation_ratio', 'device_change_flag', 
                   'location_change_flag', 'anomaly_score', 'risk_level',
                   'rule_triggered', 'final_decision']
    
    df[output_cols].to_csv(output_csv, index=False)
    print(f"\n💾 Results saved to: {output_csv}")
    
    return df, model, scaler

# ============================================
# 6. INFERENCE ON NEW DATA
# ============================================

def predict_fraud(new_data_path, model_path='fraud_model.pkl'):
    """Run inference on new transactions"""
    
    print("🔄 Loading model...")
    saved_objects = joblib.load(model_path)
    model = saved_objects['model']
    scaler = saved_objects['scaler']
    feature_cols = saved_objects['feature_cols']
    
    print("🔄 Loading new data...")
    df = load_data(new_data_path)
    
    print("🔄 Applying rules...")
    df, _ = apply_fraud_rules(df)
    
    print("🔄 Computing anomaly scores...")
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    anomaly_scores = model.decision_function(X_scaled)
    
    print("🔄 Making predictions...")
    df = make_final_decision(df, anomaly_scores)
    
    print("\n✅ Prediction complete!")
    print(f"Flagged: {(df['final_decision'] == 'FLAGGED').sum()}")
    print(f"Approved: {(df['final_decision'] == 'APPROVED').sum()}")
    
    return df

# ============================================
# RUN TRAINING
# ============================================

if __name__ == "__main__":
    # Train the model
    csv_file = "synthetic_online_transactions.csv"  # Your CSV filename
    df_results, trained_model, trained_scaler = main_training(csv_file)
    
    print("\n✅ Training complete! Model is ready for deployment.")
    print("\nTo use the model on new data, run:")
    print(">>> predictions = predict_fraud('new_transactions.csv')")
