#!/usr/bin/env python3
"""
AGGRESSIVE v5 - Exact Top Notebook Features
Based on Top Leaderboard 2 (SENet approach)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import gc

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from scipy.optimize import minimize

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
N_FOLDS = 5

DATA_DIR = Path('/home/hyzen/Kaggle Competition')
TARGET = 'exam_score'
ID_COL = 'id'

print("=" * 70)
print("🚀 AGGRESSIVE v5 - Top Notebook Features")
print("=" * 70)

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n📂 Loading datasets...")

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test.csv')
test_ids = test_df[ID_COL].values

# Original dataset
original_df = pd.read_csv(DATA_DIR / 'Exam_Score_Prediction.csv')
original_df.columns = original_df.columns.str.lower().str.replace(' ', '_')
if 'student_id' in original_df.columns:
    original_df = original_df.drop('student_id', axis=1)

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

y_train = train_df[TARGET].values
y_original = original_df[TARGET].values

# ============================================================================
# FEATURE ENGINEERING - EXACT FROM TOP NOTEBOOK 2
# ============================================================================

print("\n🔧 Feature Engineering (Top Notebook 2 Style)...")

num_features = ['study_hours', 'class_attendance', 'sleep_hours']
base_features = [col for col in train_df.columns if col not in [TARGET, 'id']]
CATS = [col for col in base_features if train_df[col].dtype == 'object']

def add_engineered_features_v2(df, train_ref=None):
    """EXACT feature engineering from Top Notebook 2"""
    df_temp = df.copy()
    
    # Sine features (EXACT from top notebook)
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32')
    
    # Log and square for numerical
    for col in num_features:
        if col in df_temp.columns:
            df_temp[f'log_{col}'] = np.log1p(df_temp[col])
            df_temp[f'{col}_sq'] = df_temp[col] ** 2
    
    # Frequency encoding for categoricals (from top notebook)
    ref_df = train_ref if train_ref is not None else df_temp
    for col in CATS:
        cat_series = df_temp[col].astype(str)
        ref_series = ref_df[col].astype(str) if train_ref is not None else cat_series
        freq_map = ref_series.value_counts().to_dict()
        df_temp[f"{col}_freq"] = cat_series.map(freq_map).fillna(0).astype(int)
    
    # THE MAGIC FORMULA (EXACT coefficients from top notebook!)
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] +
        0.34540967058057986 * df_temp['class_attendance'] +
        1.423461171860262 * df_temp['sleep_hours'] + 
        4.7819
    )
    
    # Additional interactions
    df_temp['study_x_att'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_att_ratio'] = df_temp['study_hours'] / (df_temp['class_attendance'] + 1)
    
    return df_temp

# Apply feature engineering
train_fe = add_engineered_features_v2(train_df, train_df)
test_fe = add_engineered_features_v2(test_df, train_df)
original_fe = add_engineered_features_v2(original_df, train_df)

# Identify feature columns
all_num_cols = [col for col in train_fe.columns if col not in CATS + [TARGET, 'id']]
print(f"Numerical features: {len(all_num_cols)}")

# Scale numerical features
scaler = StandardScaler()
scaler.fit(train_fe[all_num_cols])

# Ordinal encode categoricals
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(train_fe[CATS].astype(str))

def preprocess(df):
    nums = scaler.transform(df[all_num_cols])
    cats = encoder.transform(df[CATS].astype(str))
    return nums, cats

X_num, X_cat = preprocess(train_fe)
X_test_num, X_test_cat = preprocess(test_fe)
X_orig_num, X_orig_cat = preprocess(original_fe)

# Combine numerical and categorical
X_train_all = np.hstack([X_num, X_cat])
X_test_all = np.hstack([X_test_num, X_test_cat])
X_orig_all = np.hstack([X_orig_num, X_orig_cat])

print(f"X_train: {X_train_all.shape}, X_test: {X_test_all.shape}, X_orig: {X_orig_all.shape}")

# ============================================================================
# MODEL TRAINING - MULTIPLE DIVERSE MODELS
# ============================================================================

print("\n🏆 Training Multiple Models...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Storage
n_models = 4  # Ridge, LGB, XGB, CAT
oof_preds = np.zeros((len(X_train_all), n_models))
test_preds = np.zeros((len(X_test_all), n_models))
cv_scores = {i: [] for i in range(n_models)}

# Optimized params (more aggressive)
LGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.015, num_leaves=255, max_depth=12,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1
)

XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.015, max_depth=10, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
    random_state=SEED, n_jobs=-1, early_stopping_rounds=300
)

CAT_PARAMS = dict(
    iterations=3000, learning_rate=0.015, depth=10, l2_leaf_reg=3.0,
    min_data_in_leaf=20, random_seed=SEED, verbose=False, early_stopping_rounds=300
)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_all)):
    print(f"\n======== Fold {fold+1}/{N_FOLDS} ========")
    
    # BASE train/val split
    X_tr_base, X_val = X_train_all[tr_idx], X_train_all[val_idx]
    y_tr_base, y_val = y_train[tr_idx], y_train[val_idx]
    
    # AUGMENT with original data (key difference - add inside fold!)
    X_tr = np.vstack([X_tr_base, X_orig_all])
    y_tr = np.concatenate([y_tr_base, y_original])
    
    print(f"Training with {len(X_tr)} samples ({len(X_tr_base)} + {len(X_orig_all)} original)")
    
    # Model 0: Ridge
    ridge = Ridge(alpha=0.1, random_state=SEED)
    ridge.fit(X_tr, y_tr)
    pred = ridge.predict(X_val)
    oof_preds[val_idx, 0] = pred
    test_preds[:, 0] += ridge.predict(X_test_all) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    cv_scores[0].append(rmse)
    print(f"Ridge: {rmse:.5f}")
    
    # Model 1: LightGBM
    lgb_m = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(300, verbose=False)])
    pred = lgb_m.predict(X_val)
    oof_preds[val_idx, 1] = pred
    test_preds[:, 1] += lgb_m.predict(X_test_all) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    cv_scores[1].append(rmse)
    print(f"LightGBM: {rmse:.5f}")
    
    # Model 2: XGBoost
    xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred = xgb_m.predict(X_val)
    oof_preds[val_idx, 2] = pred
    test_preds[:, 2] += xgb_m.predict(X_test_all) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    cv_scores[2].append(rmse)
    print(f"XGBoost: {rmse:.5f}")
    
    # Model 3: CatBoost
    cat_m = CatBoostRegressor(**CAT_PARAMS)
    cat_m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    pred = cat_m.predict(X_val)
    oof_preds[val_idx, 3] = pred
    test_preds[:, 3] += cat_m.predict(X_test_all) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    cv_scores[3].append(rmse)
    print(f"CatBoost: {rmse:.5f}")
    
    gc.collect()

# ============================================================================
# ENSEMBLE
# ============================================================================

print("\n" + "=" * 70)
print("CV Summary:")
names = ['Ridge', 'LightGBM', 'XGBoost', 'CatBoost']
for i, name in enumerate(names):
    print(f"   {name}: {np.mean(cv_scores[i]):.5f} (+/- {np.std(cv_scores[i]):.5f})")

# Optimize weights
def rmse_obj(w):
    w = np.array(w) / np.sum(w)
    pred = (oof_preds * w).sum(axis=1)
    return np.sqrt(mean_squared_error(y_train, pred))

result = minimize(rmse_obj, [1/n_models]*n_models, bounds=[(0,1)]*n_models, method='SLSQP')
opt_w = np.array(result.x) / sum(result.x)

print("\n🔗 Optimal Weights:")
for i, name in enumerate(names):
    print(f"   {name}: {opt_w[i]:.4f}")

# Final ensemble
final_oof = (oof_preds * opt_w).sum(axis=1)
final_test = (test_preds * opt_w).sum(axis=1)

final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))
print(f"\n🏆 FINAL CV RMSE: {final_rmse:.5f}")

# ============================================================================
# STACKING (like top notebook 1)
# ============================================================================

print("\n📚 Stacking with RidgeCV...")

meta = RidgeCV(alphas=np.logspace(-2, 7, 50), scoring='neg_root_mean_squared_error')
meta.fit(oof_preds, y_train)
stack_oof = meta.predict(oof_preds)
stack_test = meta.predict(test_preds)
stack_rmse = np.sqrt(mean_squared_error(y_train, stack_oof))
print(f"   Stacking RMSE: {stack_rmse:.5f} (alpha={meta.alpha_:.2f})")

# Choose best
if stack_rmse < final_rmse:
    print("   -> Using Stacking!")
    final_test = stack_test
    final_rmse = stack_rmse
else:
    print("   -> Using Weighted Average!")

# Clip
final_test = np.clip(final_test, y_train.min(), y_train.max())

# ============================================================================
# SUBMISSION
# ============================================================================

submission = pd.DataFrame({'id': test_ids, 'exam_score': final_test})
submission.to_csv(DATA_DIR / 'submission_v5.csv', index=False)

print(f"\n✅ Saved: {DATA_DIR / 'submission_v5.csv'}")
print(f"   Mean: {final_test.mean():.2f}, Std: {final_test.std():.2f}")
print(f"   Range: [{final_test.min():.2f}, {final_test.max():.2f}]")

print("\n" + "=" * 70)
print(f"🏆 FINAL CV RMSE: {final_rmse:.5f}")
print(f"Previous best LB: 9.12360")
print("=" * 70)
