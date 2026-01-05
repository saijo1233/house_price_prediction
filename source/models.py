import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_and_evaluate(X_train, X_val, y_train, y_val):
    base_models = [
        ('ridge', Ridge(alpha=10)),
        ('lasso', Lasso(alpha=0.0005, random_state=42)),
        ('elastic', ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)),
        ('xgb', XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4, random_state=42)),
        ('lgb', LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1))
    ]
    
    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        passthrough=False 
    )
    
    models_to_test = {
        'LightGBM': base_models[6][1],
        'Stacking_Final': stacked_model
    }
    
    results = []
    best_model = None
    best_rmse = float('inf')
    
    for name, model in models_to_test.items():
        model.fit(X_train, y_train)
        
        pred_log = model.predict(X_val)
        
        pred_log = np.maximum(pred_log, 0)
        
        y_val_price = np.expm1(y_val)
        pred_price = np.expm1(pred_log)
        
        rmse_log = np.sqrt(mean_squared_error(y_val, pred_log))
        rmse_price = np.sqrt(mean_squared_error(y_val_price, pred_price))
        
        results.append({
            'Модель': name,
            'RMSE (log)': round(rmse_log, 5),
            'RMSE ($)': round(rmse_price, 0),
        })
        
        if rmse_log < best_rmse:
            best_rmse = rmse_log
            best_model = model
            
    return pd.DataFrame(results), best_model