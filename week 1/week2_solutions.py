import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('week_2_dataset.csv')

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Create feature matrix X and target y
# Price is the target column y
y = df['price']
X = df.drop('price', axis=1)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data with 20% as test data and random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Handle missing values based on instructions
# Replace missing values in the num_bathrooms feature with the Median
# Replace missing values in the house_size feature with the Mean
# Replace missing values in the house_type feature with the Most Frequent
# Replace missing values in the security_deposit feature with the Constant value 'unknown'

# Create imputers for different columns
num_bathrooms_imputer = SimpleImputer(strategy='median')
house_size_imputer = SimpleImputer(strategy='mean')
house_type_imputer = SimpleImputer(strategy='most_frequent')
security_deposit_imputer = SimpleImputer(strategy='constant', fill_value='unknown')

# Fit imputers on training data and transform both train and test
X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

# Impute num_bathrooms
if 'num_bathrooms' in X_train.columns:
    X_train_imputed['num_bathrooms'] = num_bathrooms_imputer.fit_transform(X_train[['num_bathrooms']]).ravel()
    X_test_imputed['num_bathrooms'] = num_bathrooms_imputer.transform(X_test[['num_bathrooms']]).ravel()

# Impute house_size
if 'house_size' in X_train.columns:
    X_train_imputed['house_size'] = house_size_imputer.fit_transform(X_train[['house_size']]).ravel()
    X_test_imputed['house_size'] = house_size_imputer.transform(X_test[['house_size']]).ravel()

# Impute house_type
if 'house_type' in X_train.columns:
    X_train_imputed['house_type'] = house_type_imputer.fit_transform(X_train[['house_type']]).ravel()
    X_test_imputed['house_type'] = house_type_imputer.transform(X_test[['house_type']]).ravel()

# Impute security_deposit
if 'security_deposit' in X_train.columns:
    X_train_imputed['security_deposit'] = security_deposit_imputer.fit_transform(X_train[['security_deposit']]).ravel()
    X_test_imputed['security_deposit'] = security_deposit_imputer.transform(X_test[['security_deposit']]).ravel()

# Q6: What is the mean value of house_size in the test data after imputation?
if 'house_size' in X_test_imputed.columns:
    q6_answer = X_test_imputed['house_size'].mean()
    print(f"\nQ6 - Mean value of house_size in test data after imputation: {q6_answer}")

# Q7: Apply preprocessing methods
# Ordinally Encode "room_count", "house_type", "location" features
# One-Hot Encode "city", "status", "security_deposit" features
# Scale all features using MinMaxScaler

# Define columns for different encodings
ordinal_cols = ['room_count', 'house_type', 'location']
onehot_cols = ['city', 'status', 'security_deposit']

# Filter columns that actually exist in the dataset
ordinal_cols = [col for col in ordinal_cols if col in X_train_imputed.columns]
onehot_cols = [col for col in onehot_cols if col in X_train_imputed.columns]

# Get numerical columns (excluding the ones to be encoded)
numerical_cols = X_train_imputed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = ordinal_cols + onehot_cols
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

print(f"\nOrdinal encoding columns: {ordinal_cols}")
print(f"One-hot encoding columns: {onehot_cols}")
print(f"Numerical columns: {numerical_cols}")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=1000), ordinal_cols),
        ('onehot', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), onehot_cols),
        ('num', 'passthrough', numerical_cols)
    ],
    remainder='drop'
)

# Fit preprocessor on training data and transform both sets
X_train_encoded = preprocessor.fit_transform(X_train_imputed)
X_test_encoded = preprocessor.transform(X_test_imputed)

# Scale all features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

print(f"\nScaled train shape: {X_train_scaled.shape}")
print(f"Scaled test shape: {X_test_scaled.shape}")

# Q7: Calculate the sum of all values in the first 100 rows of transformed test feature matrix
q7_answer = np.sum(X_test_scaled[:100])
print(f"\nQ7 - Sum of first 100 rows in transformed test feature matrix: {q7_answer}")

# Q8 & Q9: Create RFE + KNeighborsRegressor pipeline
# RFE with 10 features using Linear Regression as estimator
# KNeighborsRegressor with 10 neighbors

rfe_pipeline = Pipeline([
    ('rfe', RFE(estimator=LinearRegression(), n_features_to_select=10)),
    ('knn', KNeighborsRegressor(n_neighbors=10))
])

# Fit the pipeline on training data
rfe_pipeline.fit(X_train_scaled, y_train)

# Get feature names after preprocessing
feature_names = []
if ordinal_cols:
    feature_names.extend(ordinal_cols)
if onehot_cols:
    onehot_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(onehot_cols)
    feature_names.extend(onehot_feature_names)
if numerical_cols:
    feature_names.extend(numerical_cols)

print(f"\nTotal features after preprocessing: {len(feature_names)}")

# Q8: Get ranking of 'location' feature
if 'location' in ordinal_cols:
    location_idx = ordinal_cols.index('location')
    location_ranking = rfe_pipeline.named_steps['rfe'].ranking_[location_idx]
    print(f"\nQ8 - Ranking of 'location' feature by RFE: {location_ranking}")

# Q9: Calculate R2 score on test dataset
y_pred_rfe = rfe_pipeline.predict(X_test_scaled)
q9_answer = r2_score(y_test, y_pred_rfe)
print(f"\nQ9 - R2 score on test dataset: {q9_answer}")

# Q10: Create PCA + SVR pipeline with GridSearchCV
pca_svr_pipeline = Pipeline([
    ('pca', PCA()),
    ('svr', SVR())
])

# Define parameter grid
param_grid = {
    'pca__n_components': [0.9, 0.95],
    'svr__kernel': ['linear', 'poly', 'rbf']
}

# Create GridSearchCV
grid_search = GridSearchCV(
    pca_svr_pipeline,
    param_grid,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=-1
)

# Fit on training data
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Q10: Calculate RMSE on test dataset
y_pred_grid = grid_search.predict(X_test_scaled)
q10_answer = np.sqrt(mean_squared_error(y_test, y_pred_grid))
print(f"\nQ10 - Root Mean Squared Error on test dataset: {q10_answer}")

# Summary of all answers
print("\n" + "="*50)
print("SUMMARY OF ANSWERS:")
print("="*50)
if 'house_size' in X_test_imputed.columns:
    print(f"Q6 - Mean house_size in test data after imputation: {q6_answer}")
print(f"Q7 - Sum of first 100 rows in transformed test matrix: {q7_answer}")
if 'location' in ordinal_cols:
    print(f"Q8 - Ranking of 'location' feature by RFE: {location_ranking}")
print(f"Q9 - R2 score on test dataset: {q9_answer}")
print(f"Q10 - RMSE on test dataset: {q10_answer}")
