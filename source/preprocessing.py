import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def prepare_data(train_df):
    cols_to_drop = ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    train_df = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns])
    
    train_df['SalePrice_log'] = np.log1p(train_df['SalePrice'])
    
    X = train_df.drop(['Id', 'SalePrice', 'SalePrice_log'], axis=1)
    y = train_df['SalePrice_log']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preprocessor = create_preprocessor(X_train)
    
    X_train_raw = preprocessor.fit_transform(X_train)
    X_val_raw = preprocessor.transform(X_val)
    feature_names = preprocessor.get_feature_names_out()

    X_train_prep = pd.DataFrame(X_train_raw, columns=feature_names, index=X_train.index)
    X_val_prep = pd.DataFrame(X_val_raw, columns=feature_names, index=X_val.index)
    
    return X_train_prep, X_val_prep, y_train, y_val, preprocessor