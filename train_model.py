import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting ML Pipeline...")

# ─── 1. LOAD DATA ───────────────────────────────────────────────
df = pd.read_csv('train.csv')
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

raw_shape = df.shape

# ─── 2. DATA CLEANING ───────────────────────────────────────────
high_missing = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','MasVnrType']
df.drop(columns=high_missing, inplace=True)

num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print(f"✅ Cleaning done. Remaining nulls: {df.isnull().sum().sum()}")

cleaning_steps = [
    {"step": "Drop high-missing columns", "detail": f"Removed {len(high_missing)} columns with >40% missing values"},
    {"step": "Fill numeric nulls", "detail": "Replaced NaN in numeric columns with median"},
    {"step": "Fill categorical nulls", "detail": "Replaced NaN in text columns with most frequent value"},
    {"step": "Result", "detail": f"Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns"}
]

# ─── 3. FEATURE ENGINEERING ─────────────────────────────────────
df['HouseAge'] = 2024 - df['YearBuilt']
df['RemodAge'] = 2024 - df['YearRemodAdd']
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']

feature_steps = [
    {"step": "HouseAge", "detail": "2024 - YearBuilt → captures how old the house is"},
    {"step": "RemodAge", "detail": "2024 - YearRemodAdd → time since last renovation"},
    {"step": "TotalSF", "detail": "Basement + 1st Floor + 2nd Floor = total square footage"},
    {"step": "TotalBath", "detail": "Full + Half baths (weighted) for convenience score"}
]

# ─── 4. ENCODE CATEGORICALS ─────────────────────────────────────
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# ─── 5. SELECT TOP FEATURES ─────────────────────────────────────
top_features = [
    'OverallQual', 'GrLivArea', 'TotalSF', 'GarageCars',
    'TotalBsmtSF', 'HouseAge', 'TotalBath', 'LotArea',
    'YearBuilt', 'RemodAge'
]

X = df[top_features]
y = df['SalePrice']

# ─── 6. TRAIN/TEST SPLIT ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ─── 7. TRAIN RANDOM FOREST ─────────────────────────────────────
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("✅ Random Forest trained!")

# ─── 8. TRAIN LINEAR REGRESSION ─────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("✅ Linear Regression trained!")

# ─── 9. EVALUATE BOTH MODELS ────────────────────────────────────
def evaluate_model(model, X_tr, X_te, y_tr, y_te, name):
    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    acc  = round(r2 * 100, 2)
    cv   = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2')
    cv_mean = round(float(cv.mean()) * 100, 2)
    cv_std  = round(float(cv.std()) * 100, 2)
    print(f"📊 {name} → R²: {r2:.4f} | MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f}")
    return {
        "accuracy": acc,
        "r2": round(r2, 4),
        "mae": round(mae, 0),
        "rmse": round(rmse, 0),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "pred": [int(v) for v in y_pred]
    }

rf_metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")
lr_metrics = evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression")

# ─── 10. FEATURE IMPORTANCE (RF) ────────────────────────────────
importances = rf_model.feature_importances_
feat_imp = dict(zip(top_features, [round(float(i)*100, 2) for i in importances]))
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

# Linear Regression Coefficients (normalized)
coefs = lr_model.coef_
coef_abs = np.abs(coefs)
coef_pct = coef_abs / coef_abs.sum() * 100
lr_coef = dict(zip(top_features, [round(float(v), 2) for v in coef_pct]))
lr_coef_sorted = dict(sorted(lr_coef.items(), key=lambda x: x[1], reverse=True))

lr_raw_coef = dict(zip(top_features, [round(float(v), 2) for v in coefs]))

# ─── 11. CORRELATION DATA ───────────────────────────────────────
corr_features = top_features + ['SalePrice']
corr_matrix = df[corr_features].corr()['SalePrice'].drop('SalePrice').sort_values(ascending=False)
corr_data = {k: round(float(v), 3) for k, v in corr_matrix.items()}

# ─── 12. ACTUAL VS PREDICTED SAMPLE ────────────────────────────
sample_idx = np.random.choice(len(y_test), 20, replace=False)
actual_sample  = [int(v) for v in y_test.iloc[sample_idx].values]
rf_pred_sample = [int(v) for v in np.array(rf_metrics["pred"])[sample_idx]]
lr_pred_sample = [int(v) for v in np.array(lr_metrics["pred"])[sample_idx]]

# ─── 13. SAVE MODELS ────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('lr_model.pkl', 'wb') as f:
    pickle.dump({'model': lr_model, 'scaler': scaler}, f)

# ─── 14. SAVE METRICS ────────────────────────────────────────────
metrics = {
    # Common
    "train_size": len(X_train),
    "test_size": len(X_test),
    "raw_shape": list(raw_shape),
    "top_features": top_features,
    "correlation": corr_data,
    "actual_sample": actual_sample,
    "cleaning_steps": cleaning_steps,
    "feature_steps": feature_steps,

    # Random Forest
    "rf": {
        "algorithm": "Random Forest Regressor",
        "n_estimators": 200,
        "accuracy": rf_metrics["accuracy"],
        "r2": rf_metrics["r2"],
        "mae": rf_metrics["mae"],
        "rmse": rf_metrics["rmse"],
        "cv_mean": rf_metrics["cv_mean"],
        "cv_std": rf_metrics["cv_std"],
        "feature_importance": feat_imp_sorted,
        "pred_sample": rf_pred_sample,
    },

    # Linear Regression
    "lr": {
        "algorithm": "Linear Regression",
        "accuracy": lr_metrics["accuracy"],
        "r2": lr_metrics["r2"],
        "mae": lr_metrics["mae"],
        "rmse": lr_metrics["rmse"],
        "cv_mean": lr_metrics["cv_mean"],
        "cv_std": lr_metrics["cv_std"],
        "feature_importance": lr_coef_sorted,
        "raw_coefficients": lr_raw_coef,
        "pred_sample": lr_pred_sample,
    },

    # Legacy (for index.html compat)
    "accuracy": rf_metrics["accuracy"],
    "r2": rf_metrics["r2"],
    "mae": rf_metrics["mae"],
    "algorithm": "Random Forest Regressor",
    "n_estimators": 200,
    "feature_importance": feat_imp_sorted,
    "pred_sample": rf_pred_sample,
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✅ model.pkl, lr_model.pkl and metrics.json saved!")
print(f"\n🎯 Random Forest Accuracy: {rf_metrics['accuracy']}%")
print(f"🎯 Linear Regression Accuracy: {lr_metrics['accuracy']}%")
