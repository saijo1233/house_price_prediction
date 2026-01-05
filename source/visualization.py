import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor

def plot_regression_analysis(y_val, pred_val, model_name="Stacking Model"):
    real_price = np.expm1(y_val)
    pred_price = np.expm1(pred_val)
    residuals = real_price - pred_price

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x=real_price, y=pred_price, alpha=0.6, ax=ax[0], color='teal')
    ax[0].plot([real_price.min(), real_price.max()], [real_price.min(), real_price.max()], 
               'r--', lw=2)
    ax[0].set_title(f'{model_name}: Реальные vs Предсказанные цены', fontsize=14)
    ax[0].set_xlabel('Реальная цена ($)')
    ax[0].set_ylabel('Предсказанная цена ($)')

    sns.histplot(residuals, kde=True, ax=ax[1], color='indianred')
    ax[1].set_title('Распределение остатков (Ошибок)', fontsize=14)
    ax[1].set_xlabel('Ошибка ($)')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, model_name="Модель", top_n=20):
    importances = None
    actual_model_name = model_name

    if isinstance(model, StackingRegressor):
        for name, est in model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                actual_model_name = f"{model_name} (Base: {name})"
                break
    
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    
    if importances is None:
        print(f"Модель {type(model).__name__} не поддерживает визуализацию важности напрямую.")
        return

    # Создаем DataFrame
    importance_df = pd.DataFrame({
        'Признак': feature_names,
        'Важность': importances
    }).sort_values(by='Важность', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    
    sns.barplot(
        x='Важность', 
        y='Признак', 
        data=importance_df, 
        palette='viridis',
        hue='Признак',    # Присваиваем y переменной hue
        legend=False      # Отключаем легенду, так как она здесь не нужна
    )
    
    plt.title(f'Топ-{top_n} важных признаков\n{actual_model_name}', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_price_distribution(df, column='SalePrice'):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(df[column], kde=True, ax=ax[0], color='blue')
    ax[0].set_title(f'Распределение {column} (Original)', fontsize=14)

    sns.histplot(np.log1p(df[column]), kde=True, ax=ax[1], color='green')
    ax[1].set_title(f'Распределение {column} (Log Transformed)', fontsize=14)
    
    plt.show()