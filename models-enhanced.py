#ip file
from google.colab import files
uploaded = files.upload()

# === Enhanced Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import (
    chi2, SelectKBest, mutual_info_classif, f_classif, RFE
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
    has_lightgbm = True
except ImportError:
    has_lightgbm = False

# === Step 1: Load Data ===
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)
target_col = 'next_month_default'

# === Step 2: Data Validation and Cleaning ===
selected_features = [
    'LIMIT_BAL', 'age', 'AVG_Bill_amt', 'PAY_TO_BILL_ratio',
    'pay_0', 'pay_2', 'pay_3', 'pay_amt1', 'pay_amt2', 'pay_amt3'
]

# Validate features exist
for col in selected_features + [target_col]:
    if col not in df.columns:
        raise ValueError(f"❌ Required column '{col}' not found in uploaded file.")

X = df[selected_features].copy()
y = df[target_col].copy()

print("🔍 Dataset Overview:")
print(f"Samples: {len(X)}")
print(f"Features: {len(selected_features)}")
print(f"Class 0: {len(y[y==0])} ({len(y[y==0])/len(y)*100:.1f}%)")
print(f"Class 1: {len(y[y==1])} ({len(y[y==1])/len(y)*100:.1f}%)")

# Check for missing values
print(f"\n🔍 Missing Value Analysis:")
missing_counts = X.isnull().sum()
print("Missing values per feature:")
for feature, count in missing_counts.items():
    if count > 0:
        print(f"  {feature}: {count} ({count/len(X)*100:.1f}%)")

# === Step 3: Handle Missing Values First ===
print("\n🔧 Preprocessing: Handling Missing Values...")
# Create a simple imputer to handle NaN values before feature selection
initial_imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(
    initial_imputer.fit_transform(X), 
    columns=X.columns, 
    index=X.index
)

print("✅ Missing values handled successfully")

# === Step 4: Enhanced Feature Statistics ===
def calculate_feature_statistics(X, y):
    """Calculate comprehensive feature statistics"""
    feature_stats = {}
    
    for i, feature in enumerate(X.columns):
        feature_data = X[feature].values
        
        # Basic statistics
        stats_dict = {
            'mean': np.mean(feature_data),
            'std': np.std(feature_data),
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'skewness': skew(feature_data),
            'kurtosis': kurtosis(feature_data),
        }
        
        # Chi-squared test (handle potential issues)
        try:
            # Ensure no negative values for chi2 test
            feature_positive = feature_data - np.min(feature_data) + 0.001
            chi2_score, p_val = chi2(feature_positive.reshape(-1, 1), y)
            stats_dict['chi2_score'] = chi2_score[0]
            stats_dict['chi2_p_value'] = p_val[0]
        except Exception as e:
            stats_dict['chi2_score'] = 0
            stats_dict['chi2_p_value'] = 1
            
        # Mutual Information
        try:
            mi_score = mutual_info_classif(feature_data.reshape(-1, 1), y, random_state=42)
            stats_dict['mutual_info'] = mi_score[0]
        except Exception as e:
            stats_dict['mutual_info'] = 0
            
        # F-statistic
        try:
            f_score, f_p_val = f_classif(feature_data.reshape(-1, 1), y)
            stats_dict['f_score'] = f_score[0]
            stats_dict['f_p_value'] = f_p_val[0]
        except Exception as e:
            stats_dict['f_score'] = 0
            stats_dict['f_p_value'] = 1
            
        feature_stats[feature] = stats_dict
    
    return feature_stats

print("\n📊 Computing Feature Statistics...")
feature_statistics = calculate_feature_statistics(X_imputed, y)

# Display top features by different criteria
print("\n🏆 Top Features by Mutual Information:")
mi_sorted = sorted(feature_statistics.items(), 
                   key=lambda x: x[1]['mutual_info'], reverse=True)
for feature, stats in mi_sorted[:5]:
    print(f"  {feature}: {stats['mutual_info']:.4f}")

print("\n🏆 Top Features by F-Score:")
f_sorted = sorted(feature_statistics.items(), 
                  key=lambda x: x[1]['f_score'], reverse=True)
for feature, stats in f_sorted[:5]:
    print(f"  {feature}: {stats['f_score']:.4f}")

# === Step 5: Robust Feature Selection Methods ===
def apply_feature_selection_methods(X, y, n_features=8):
    """Apply multiple feature selection methods with proper error handling"""
    feature_selection_results = {}
    
    # Ensure data is properly formatted
    X_array = X.values if hasattr(X, 'values') else X
    feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X_array.shape[1])]
    
    # Chi-squared selection (handle negative values)
    try:
        # Make all values non-negative for chi2 test
        X_chi2_prep = X_array - np.min(X_array, axis=0) + 0.001
        chi2_selector = SelectKBest(chi2, k=min(n_features, X_array.shape[1]))
        chi2_selector.fit(X_chi2_prep, y)
        chi2_features = [feature_names[i] for i in chi2_selector.get_support(indices=True)]
        feature_selection_results['chi2'] = chi2_features
    except Exception as e:
        print(f"⚠️ Chi-squared selection failed: {e}")
        feature_selection_results['chi2'] = list(feature_names[:n_features])
    
    # Mutual Information selection
    try:
        mi_selector = SelectKBest(mutual_info_classif, k=min(n_features, X_array.shape[1]))
        mi_selector.fit(X_array, y)
        mi_features = [feature_names[i] for i in mi_selector.get_support(indices=True)]
        feature_selection_results['mutual_info'] = mi_features
    except Exception as e:
        print(f"⚠️ Mutual Information selection failed: {e}")
        feature_selection_results['mutual_info'] = list(feature_names[:n_features])
    
    # F-statistic selection
    try:
        f_selector = SelectKBest(f_classif, k=min(n_features, X_array.shape[1]))
        f_selector.fit(X_array, y)
        f_features = [feature_names[i] for i in f_selector.get_support(indices=True)]
        feature_selection_results['f_statistic'] = f_features
    except Exception as e:
        print(f"⚠️ F-statistic selection failed: {e}")
        feature_selection_results['f_statistic'] = list(feature_names[:n_features])
    
    # Random Forest importance
    try:
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X_array, y)
        importance_scores = rf_temp.feature_importances_
        top_indices = np.argsort(importance_scores)[-n_features:]
        rf_features = [feature_names[i] for i in top_indices]
        feature_selection_results['random_forest'] = rf_features
    except Exception as e:
        print(f"⚠️ Random Forest selection failed: {e}")
        feature_selection_results['random_forest'] = list(feature_names[:n_features])
    
    # Recursive Feature Elimination
    try:
        rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=min(n_features, X_array.shape[1]))
        rfe_selector.fit(X_array, y)
        rfe_features = [feature_names[i] for i in rfe_selector.get_support(indices=True)]
        feature_selection_results['rfe'] = rfe_features
    except Exception as e:
        print(f"⚠️ RFE selection failed: {e}")
        feature_selection_results['rfe'] = list(feature_names[:n_features])
    
    return feature_selection_results

print("\n🎯 Applying Feature Selection Methods...")
feature_selection_results = apply_feature_selection_methods(X_imputed, y, n_features=8)

for method, features in feature_selection_results.items():
    print(f"\n{method.upper()} Selected Features: {features}")

# Use the method with best theoretical backing (mutual information for non-linear relationships)
selected_optimal_features = feature_selection_results['mutual_info']
X_selected = X_imputed[selected_optimal_features].copy()

print(f"\n✅ Using {len(selected_optimal_features)} optimally selected features")
print(f"Selected features: {selected_optimal_features}")

# === Step 6: Advanced Preprocessing Pipeline ===
def create_preprocessing_pipelines():
    """Create multiple preprocessing pipelines"""
    pipelines = {
        'standard': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]),
        'minmax': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ]),
        'robust': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', RobustScaler())
        ]),
        'simple': SimpleImputer(strategy='mean')
    }
    return pipelines

preprocessing_pipelines = create_preprocessing_pipelines()

# === Step 7: Advanced Sampling Strategies ===
def apply_sampling_strategy(X, y, strategy='smote'):
    """Apply advanced sampling strategies with error handling"""
    try:
        if strategy == 'smote':
            sampler = SMOTE(random_state=42)
        elif strategy == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        else:
            return X, y
        
        return sampler.fit_resample(X, y)
    except Exception as e:
        print(f"⚠️ Sampling strategy {strategy} failed: {e}")
        print("Using original data without resampling")
        return X, y

# === Step 8: Enhanced Model Definition ===
def create_enhanced_models():
    """Create enhanced model pipelines with different preprocessing"""
    models = {}
    
    # Random Forest with different preprocessing
    models['Random Forest (Standard)'] = Pipeline([
        ('prep', preprocessing_pipelines['standard']),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    
    models['Random Forest (Robust)'] = Pipeline([
        ('prep', preprocessing_pipelines['robust']),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    
    # XGBoost with enhanced parameters
    models['XGBoost (Optimized)'] = Pipeline([
        ('prep', preprocessing_pipelines['simple']),
        ('clf', XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1
        ))
    ])
    
    # SVM with different kernels
    models['SVM (RBF)'] = Pipeline([
        ('prep', preprocessing_pipelines['standard']),
        ('clf', SVC(
            probability=True, class_weight='balanced',
            kernel='rbf', C=1.0, gamma='scale', random_state=42
        ))
    ])
    
    # Gradient Boosting
    models['Gradient Boosting'] = Pipeline([
        ('prep', preprocessing_pipelines['simple']),
        ('clf', GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42
        ))
    ])
    
    # LightGBM if available
    if has_lightgbm:
        models['LightGBM'] = Pipeline([
            ('prep', preprocessing_pipelines['simple']),
            ('clf', LGBMClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.1,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1
            ))
        ])
    
    return models

models = create_enhanced_models()

# === Step 9: Stratified K-Fold Cross-Validation ===
def perform_cross_validation_evaluation(models, X, y, cv_folds=5):
    """Perform comprehensive cross-validation evaluation"""
    print(f"\n🔄 Performing {cv_folds}-Fold Stratified Cross-Validation...")
    
    # Apply SMOTE sampling
    print("⚖️ Applying SMOTE for class balancing...")
    X_resampled, y_resampled = apply_sampling_strategy(X, y, 'smote')
    
    class_counts = np.bincount(y_resampled)
    print(f"After SMOTE - Samples: {len(X_resampled)}, Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    
    cv_results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n🔁 Evaluating {name}...")
        
        try:
            # Cross-validation scoring
            cv_scores = {
                'accuracy': cross_val_score(model, X_resampled, y_resampled, 
                                          cv=skf, scoring='accuracy', n_jobs=-1),
                'precision': cross_val_score(model, X_resampled, y_resampled, 
                                           cv=skf, scoring='precision', n_jobs=-1),
                'recall': cross_val_score(model, X_resampled, y_resampled, 
                                        cv=skf, scoring='recall', n_jobs=-1),
                'f1': cross_val_score(model, X_resampled, y_resampled, 
                                    cv=skf, scoring='f1', n_jobs=-1),
                'roc_auc': cross_val_score(model, X_resampled, y_resampled, 
                                         cv=skf, scoring='roc_auc', n_jobs=-1)
            }
            
            # Calculate statistics
            cv_results[name] = {
                'accuracy_mean': cv_scores['accuracy'].mean(),
                'accuracy_std': cv_scores['accuracy'].std(),
                'precision_mean': cv_scores['precision'].mean(),
                'precision_std': cv_scores['precision'].std(),
                'recall_mean': cv_scores['recall'].mean(),
                'recall_std': cv_scores['recall'].std(),
                'f1_mean': cv_scores['f1'].mean(),
                'f1_std': cv_scores['f1'].std(),
                'roc_auc_mean': cv_scores['roc_auc'].mean(),
                'roc_auc_std': cv_scores['roc_auc'].std()
            }
            
        except Exception as e:
            print(f"⚠️ Error evaluating {name}: {e}")
            # Provide default values to prevent crash
            cv_results[name] = {
                'accuracy_mean': 0.5, 'accuracy_std': 0.0,
                'precision_mean': 0.5, 'precision_std': 0.0,
                'recall_mean': 0.5, 'recall_std': 0.0,
                'f1_mean': 0.5, 'f1_std': 0.0,
                'roc_auc_mean': 0.5, 'roc_auc_std': 0.0
            }
    
    return cv_results, X_resampled, y_resampled

# Perform cross-validation evaluation
cv_results, X_resampled, y_resampled = perform_cross_validation_evaluation(
    models, X_selected, y, cv_folds=5
)

# === Step 10: Enhanced Results Display ===
print("\n" + "="*80)
print("📈 COMPREHENSIVE MODEL EVALUATION RESULTS")
print("="*80)

# Create results DataFrame
results_data = []
for model_name, results in cv_results.items():
    results_data.append({
        'Model': model_name,
        'Accuracy': f"{results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}",
        'Precision': f"{results['precision_mean']:.4f} ± {results['precision_std']:.4f}",
        'Recall': f"{results['recall_mean']:.4f} ± {results['recall_std']:.4f}",
        'F1 Score': f"{results['f1_mean']:.4f} ± {results['f1_std']:.4f}",
        'ROC AUC': f"{results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}",
        'F1_numeric': results['f1_mean']  
    })

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values('F1_numeric', ascending=False)
results_df = results_df.drop('F1_numeric', axis=1)

print(results_df.set_index('Model').to_string())

# === Step 11: Best Model Analysis ===
best_model_name = max(cv_results.keys(), 
                     key=lambda x: cv_results[x]['f1_mean'])
best_model = models[best_model_name]

print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
print(f"F1 Score: {cv_results[best_model_name]['f1_mean']:.4f} ± {cv_results[best_model_name]['f1_std']:.4f}")
print(f"ROC AUC: {cv_results[best_model_name]['roc_auc_mean']:.4f} ± {cv_results[best_model_name]['roc_auc_std']:.4f}")

# === Step 12: Enhanced Voting Classifier ===
print("\n🗳️ Creating Enhanced Voting Classifier...")

# Select top 3 models for voting (ensure odd number)
top_models = [(name, models[name]) for name in 
              sorted(cv_results.keys(), 
                    key=lambda x: cv_results[x]['f1_mean'], 
                    reverse=True)[:3]]

print(f"Selected models for voting: {[name for name, _ in top_models]}")

try:
    voting_clf = VotingClassifier(estimators=top_models, voting='soft', n_jobs=-1)
    
    # Evaluate voting classifier
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    voting_cv_scores = {
        'accuracy': cross_val_score(voting_clf, X_resampled, y_resampled, 
                                   cv=skf, scoring='accuracy', n_jobs=-1),
        'f1': cross_val_score(voting_clf, X_resampled, y_resampled, 
                             cv=skf, scoring='f1', n_jobs=-1),
        'roc_auc': cross_val_score(voting_clf, X_resampled, y_resampled, 
                                  cv=skf, scoring='roc_auc', n_jobs=-1)
    }
    
    print(f"\n✅ VOTING CLASSIFIER RESULTS:")
    print(f"Accuracy: {voting_cv_scores['accuracy'].mean():.4f} ± {voting_cv_scores['accuracy'].std():.4f}")
    print(f"F1 Score: {voting_cv_scores['f1'].mean():.4f} ± {voting_cv_scores['f1'].std():.4f}")
    print(f"ROC AUC: {voting_cv_scores['roc_auc'].mean():.4f} ± {voting_cv_scores['roc_auc'].std():.4f}")
    
except Exception as e:
    print(f"⚠️ Voting classifier evaluation failed: {e}")

# === Step 13: Single Train-Test Split Validation ===
print("\n📊 Single Train-Test Split Validation...")

# Split the original selected data for final validation
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE to training data only
X_train_resampled, y_train_resampled = apply_sampling_strategy(X_train, y_train, 'smote')

# Train and evaluate the best model
print(f"Training {best_model_name} on train-test split...")
best_model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_prob)

print(f"\n🎯 FINAL TEST SET RESULTS ({best_model_name}):")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")
print(f"ROC AUC  : {test_auc:.4f}")

# === Step 14: Feature Importance Analysis ===
print("\n🎯 Feature Importance Analysis...")
try:
    if 'Random Forest' in best_model_name or 'XGBoost' in best_model_name or 'Gradient Boosting' in best_model_name:
        if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
            importances = best_model.named_steps['clf'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': selected_optimal_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n🔥 Top Feature Importances:")
            for idx, row in feature_importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
except Exception as e:
    print(f"⚠️ Feature importance analysis failed: {e}")

# === Step 15: Visualization ===
try:
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {test_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Final Test Set')
    plt.legend()
    plt.grid(True)
    
    # Confusion Matrix
    plt.subplot(1, 2, 2)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'Confusion Matrix\n{best_model_name}')
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"⚠️ Visualization failed: {e}")

print("\n" + "="*80)
print("🎉 ENHANCED ANALYSIS COMPLETE!")
print("="*80)
print("\n📝 SUMMARY:")
print(f"✅ Dataset processed: {len(X)} samples, {len(selected_optimal_features)} selected features")
print(f"✅ Best model: {best_model_name}")
print(f"✅ Cross-validation F1: {cv_results[best_model_name]['f1_mean']:.4f}")
print(f"✅ Test set F1: {test_f1:.4f}")
print(f"✅ All models evaluated with robust error handling")
