"""
Synthetic Fraud Transaction Data Generator
Generates 10,000 realistic transactions for training fraud detection model
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.1):
    """
    Generate synthetic transaction data
    
    Args:
        n_samples: Total number of transactions (default: 10,000)
        fraud_ratio: Percentage of fraud transactions (default: 0.1 = 10%)
    
    Returns:
        DataFrame with features and labels
    """
    
    print(f"🔧 Generating {n_samples} synthetic transactions...")
    print(f"   - Legitimate: {int(n_samples * (1 - fraud_ratio))}")
    print(f"   - Fraud: {int(n_samples * fraud_ratio)}")
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    data = []
    
    # Generate LEGITIMATE transactions (90%)
    print("\n📊 Generating legitimate transactions...")
    for i in range(n_legitimate):
        # Normal business hours (6 AM - 11 PM, including 23)
        hour = random.choice([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
        
        # Typical investment amounts (₹500 - ₹5,000)
        amount = random.uniform(500, 5000)
        
        # Weekday more common
        is_weekend = random.random() < 0.2  # 20% on weekends
        
        # Amount vs average (close to 1.0 for normal)
        amount_vs_avg = random.uniform(0.5, 1.5)
        
        # Time since last transaction (hours) - regular pattern
        time_since_last = random.uniform(1, 48)
        
        # Same location/device (95% of time)
        location_change = random.random() < 0.05
        device_change = random.random() < 0.05
        
        # No failed logins
        failed_logins = random.choice([0, 0, 0, 1])
        
        # Established account (months)
        account_age = random.uniform(6, 60)
        
        # Low historical risk
        historical_risk = random.uniform(0.0, 0.2)
        
        data.append({
            'amount': amount,
            'hour': hour,
            'is_weekend': int(is_weekend),
            'amount_vs_avg': amount_vs_avg,
            'time_since_last': time_since_last,
            'location_change': int(location_change),
            'device_change': int(device_change),
            'failed_logins': failed_logins,
            'account_age': account_age,
            'historical_risk': historical_risk,
            'is_fraud': 0
        })
    
    # Generate FRAUD transactions (10%)
    print("🚨 Generating fraud transactions...")
    for i in range(n_fraud):
        # Odd hours (midnight to 5 AM) - NOT including 23
        hour = random.choice([0, 1, 2, 3, 4, 5])
        
        # Large amounts (₹20,000 - ₹100,000)
        amount = random.uniform(20000, 100000)
        
        # Any day
        is_weekend = random.random() < 0.5
        
        # Amount much higher than average
        amount_vs_avg = random.uniform(5.0, 20.0)
        
        # Quick succession or very long gap
        time_since_last = random.choice([
            random.uniform(0.1, 0.5),  # Very quick (minutes)
            random.uniform(100, 500)   # Very long (days)
        ])
        
        # Location/device changes (80% of time)
        location_change = random.random() < 0.8
        device_change = random.random() < 0.8
        
        # Failed login attempts
        failed_logins = random.choice([2, 3, 4, 5])
        
        # Newer accounts or very old compromised
        account_age = random.choice([
            random.uniform(0.1, 3),    # New account
            random.uniform(24, 60)     # Old compromised
        ])
        
        # High historical risk
        historical_risk = random.uniform(0.5, 1.0)
        
        data.append({
            'amount': amount,
            'hour': hour,
            'is_weekend': int(is_weekend),
            'amount_vs_avg': amount_vs_avg,
            'time_since_last': time_since_last,
            'location_change': int(location_change),
            'device_change': int(device_change),
            'failed_logins': failed_logins,
            'account_age': account_age,
            'historical_risk': historical_risk,
            'is_fraud': 1
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Generated {len(df)} transactions")
    print(f"\n📊 Feature Statistics:")
    print(df.describe())
    
    return df

def save_data(df, filepath='data/training_data/fraud_transactions.csv'):
    """Save synthetic data to CSV"""
    df.to_csv(filepath, index=False)
    print(f"\n💾 Data saved to: {filepath}")

if __name__ == '__main__':
    # Generate data
    df = generate_synthetic_data(n_samples=10000, fraud_ratio=0.1)
    
    # Save to file
    save_data(df)
    
    print("\n✅ Synthetic data generation complete!")
