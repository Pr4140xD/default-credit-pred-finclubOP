              # DOING EDA ON THE DATASET PART--1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def load_data():
    if IN_COLAB:
        uploaded = files.upload()
        file_name = next(iter(uploaded))
        df = pd.read_csv(file_name)
    else:
        df = pd.read_csv('train_dataset_final1.csv')
    df.columns = df.columns.str.strip().str.lower()
    return df

# Load data
df = load_data()
print("✓ Data loaded successfully")
print("Columns:", df.columns.tolist())
print("\n=== Initial Data Summary ===")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# Target variable analysis
if 'next_month_default' in df.columns:
    plt.figure(figsize=(8,5))
    sns.countplot(x='next_month_default', data=df)
    plt.title('Class Distribution (Default vs Non-Default)')
    plt.show()
    print("\nClass balance:\n", df['next_month_default'].value_counts(normalize=True))

# ========== Payment Behavior Analysis ==========
print("\n=== Payment Pattern Analysis ===")
pay_cols = [f'pay_{i}' for i in [0,2,3,4,5,6]]  

if all(col in df.columns for col in pay_cols):
  
    df[pay_cols] = df[pay_cols].replace({-2: -1})  
    
    # Delinquency metrics
    df['delinquency_streak'] = df[pay_cols].apply(lambda x: (x > 0).sum(), axis=1)
    df['max_consecutive_delinquency'] = df[pay_cols].apply(
        lambda x: max((x > 0).astype(int).cumsum().diff().eq(1).cumsum()), axis=1)
    
    # Payment status evolution
    plt.figure(figsize=(12,6))
    payment_status = df[pay_cols].apply(pd.Series.value_counts, normalize=True).T
    sns.heatmap(payment_status, annot=True, fmt=".1%", cmap="YlGnBu")
    plt.title('Payment Status Distribution Over 6 Months')
    plt.xlabel('Payment Status Code')
    plt.ylabel('Month (0 = Recent)')
    plt.show()
else:
    print("Skipping payment analysis - missing columns")

# ========== Credit Utilization Analysis ==========
print("\n=== Credit Utilization Patterns ===")
bill_cols = [f'bill_amt{i}' for i in range(1,7)]

if all(col in df.columns for col in bill_cols) and 'limit_bal' in df.columns:
    # Handle zero credit limits
    df['limit_bal'] = df['limit_bal'].replace(0, 1e-6)
    
    
    df['avg_utilization'] = df[bill_cols].mean(axis=1) / df['limit_bal']
    df['max_utilization'] = df[bill_cols].max(axis=1) / df['limit_bal']
    
    # Utilization categories (financial standards)
    bins = [-np.inf, 0.1, 0.3, 0.5, 0.7, 1.0, np.inf]
    labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Over Limit']
    df['utilization_category'] = pd.cut(df['avg_utilization'], bins=bins, labels=labels)
    
    # Utilization trend analysis
    df['utilization_trend'] = df[bill_cols].apply(
        lambda row: np.polyfit(range(6), row/df.loc[row.name, 'limit_bal'], 1)[0], 
        axis=1
    )
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.boxplot(x='next_month_default', y='avg_utilization', data=df)
    plt.title('Average Utilization vs Default')
    
    plt.subplot(1,2,2)
    sns.countplot(x='utilization_category', hue='next_month_default', data=df, order=labels)
    plt.title('Default Distribution by Utilization Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping utilization analysis - missing columns")

# ========== Repayment Consistency Analysis ==========
print("\n=== Repayment Behavior ===")
pay_amt_cols = [f'pay_amt{i}' for i in range(1,7)]

if all(col in df.columns for col in bill_cols + pay_amt_cols):
    df['total_payment'] = df[pay_amt_cols].sum(axis=1)
    df['total_bill'] = df[bill_cols].sum(axis=1)
    df['payment_deficit'] = df['total_bill'] - df['total_payment']
    df['payment_ratio'] = df['total_payment'] / (df['total_bill'] + 1e-6)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(data=df, x='payment_ratio', hue='next_month_default', bins=30, kde=True)
    plt.title('Payment Ratio Distribution')
    
    plt.subplot(1,2,2)
    sns.scatterplot(data=df, x='total_bill', y='total_payment', hue='next_month_default', alpha=0.5)
    plt.title('Bill Amount vs Payment Amount')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping repayment analysis - missing columns")

# ========== Demographic Analysis ==========
print("\n=== Demographic Risk Factors ===")
demographic_cols = ['marriage', 'sex', 'education', 'age', 'limit_bal']

if all(col in df.columns for col in demographic_cols):
    # Education mapping
    edu_mapping = {1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Others', 0: 'Unknown'}
    df['education'] = df['education'].map(edu_mapping)
    
    # Marriage mapping
    marriage_mapping = {1: 'Married', 2: 'Single', 3: 'Divorce', 0: 'Unknown'}
    df['marriage'] = df['marriage'].map(marriage_mapping)
    
    # Age bins
    df['age_group'] = pd.cut(df['age'], bins=[20,30,40,50,60,80], 
                            labels=['20-29', '30-39', '40-49', '50-59', '60+'])
    
    # Visualization
    fig, ax = plt.subplots(2, 2, figsize=(15,12))
    
    sns.countplot(x='education', hue='next_month_default', data=df, ax=ax[0,0])
    ax[0,0].set_title('Education Level vs Default')
    
    sns.countplot(x='age_group', hue='next_month_default', data=df, ax=ax[0,1])
    ax[0,1].set_title('Age Group vs Default')
    
    sns.boxplot(x='next_month_default', y='limit_bal', data=df, ax=ax[1,0])
    ax[1,0].set_yscale('log')
    ax[1,0].set_title('Credit Limit vs Default')
    
    sns.countplot(x='marriage', hue='next_month_default', data=df, ax=ax[1,1])
    ax[1,1].set_title('Marital Status vs Default')
    
    plt.tight_layout()
    plt.show()
else:
    print("Skipping demographic analysis - missing columns")

# ========== Feature Correlation Analysis ==========
print("\n=== Feature Correlations ===")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(16,12))
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', center=0, 
            mask=np.triu(np.ones_like(df[numeric_cols].corr(), dtype=bool)))
plt.title('Feature Correlation Matrix')
plt.show()

# ========== Save Processed Data ==========
output_file = 'processed_credit_data.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ EDA complete! Processed data saved to {output_file}")
print("New features created:", list(df.columns[-10:]))
        #=====DOING EDA PART-2 =========#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def load_data():
    if IN_COLAB:
        print("Upload your CSV file:")
        uploaded = files.upload()
        file_name = next(iter(uploaded))
        return pd.read_csv(file_name)
    else:
        return pd.read_csv('your_train_data.csv')  # Change as needed

# Load and clean data
df = load_data()
df.columns = df.columns.str.strip().str.lower()
print("Columns:", df.columns.tolist())

# --- Basic Info ---
print("\n=== Dataset Overview ===")
print("Shape:", df.shape)
print("First 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# --- Target Variable ---
target_col = 'next_month_default'
if target_col in df.columns:
    print("\nTarget Distribution:")
    print(df[target_col].value_counts())
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title('Target Distribution')
    plt.show()

# --- Feature Statistics ---
print("\n=== Feature Statistics ===")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    print(f"\n--- {col} ---")
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Mean:", df[col].mean())
    print("Std:", df[col].std())
    print("Variance:", df[col].var())
    print("Skewness:", stats.skew(df[col].dropna()))
    print("Kurtosis:", stats.kurtosis(df[col].dropna(), fisher=True))

# --- Distribution Plots ---
for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

# --- Correlation Matrix ---
plt.figure(figsize=(12,10))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# --- Payment Behavior (if columns exist) ---
pay_cols = [col for col in df.columns if col.startswith('pay_')]
if pay_cols:
    print("\nCalculating delinquency streak using columns:", pay_cols)
    df['delinquency_streak'] = df[pay_cols].apply(lambda x: (x >= 1).sum(), axis=1)
    plt.figure(figsize=(8,4))
    sns.histplot(df['delinquency_streak'], bins=len(pay_cols)+1)
    plt.title('Delinquency Streak Distribution')
    plt.show()
else:
    print("No payment status columns found for delinquency analysis.")

# --- Utilization (if columns exist) ---
bill_cols = [col for col in df.columns if col.startswith('bill_amt')]
if bill_cols and 'limit_bal' in df.columns:
    df['avg_utilization'] = df[bill_cols].mean(axis=1) / df['limit_bal']
    bins = [-np.inf, 0.1, 0.3, 0.5, 0.7, 1.0, np.inf]
    labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Over Limit']
    df['utilization_category'] = pd.cut(df['avg_utilization'], bins=bins, labels=labels)
    plt.figure(figsize=(10,4))
    if target_col in df.columns:
        sns.countplot(x='utilization_category', hue=target_col, data=df, order=labels)
    else:
        sns.countplot(x='utilization_category', data=df, order=labels)
    plt.title('Utilization Category Distribution')
    plt.show()
else:
    print("Insufficient columns for utilization analysis.")

# --- Save processed data ---
df.to_csv('processed_training_data.csv', index=False)
print("\nEDA complete. Processed data saved as 'processed_training_data.csv'.")

