import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import pandas as pd

def train_and_evaluate(X_train, X_val, y_train, y_val):
    models = {
        'LightGBM': lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, 
                                      num_leaves=31, random_state=42),
        'Ridge': Ridge(alpha=10),
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    
    results = []
    best_model = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        
        rmse_log = np.sqrt(mean_squared_error(y_val, pred))
        rmse_price = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(pred)))
        mae_price = mean_absolute_error(np.expm1(y_val), np.expm1(pred))
        
        results.append({
            'Модель': name,
            'RMSE (log)': round(rmse_log, 5),
            'RMSE ($)': round(rmse_price, 0),
            'MAE ($)': round(mae_price, 0)
        })
        
        if rmse_log < best_rmse:
            best_rmse = rmse_log
            best_model = model
    
    return pd.DataFrame(results), best_model