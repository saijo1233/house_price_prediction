import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_predictions(y_val, pred_val, model_name="Модель"):
    """
    Визуализирует реальные vs предсказанные цены на валидационной выборке.
    
    Parameters:
    y_val (pd.Series or np.array): Реальные значения (в логарифмированном масштабе).
    pred_val (np.array): Предсказанные значения (в логарифмированном масштабе).
    model_name (str): Название модели для заголовка графика.
    """
    real_price = np.expm1(y_val)      # Обратный логарифм — реальные цены в $
    pred_price = np.expm1(pred_val)   # Предсказанные цены в $

    plt.figure(figsize=(10, 8))
    plt.scatter(real_price, pred_price, alpha=0.6, color='steelblue', edgecolor='k')
    min_val = min(real_price.min(), pred_price.min())
    max_val = max(real_price.max(), pred_price.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальная линия')
    plt.xlabel('Реальная цена ($)', fontsize=12)
    plt.ylabel('Предсказанная цена ($)', fontsize=12)
    plt.title(f'Реальные vs Предсказанные цены\nЛучшая модель: {model_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, model_name="Модель", top_n=20):
    """
    Визуализирует топ-N важных признаков в зависимости от типа модели.
    
    Parameters:
    model: Обученная модель (Ridge, LightGBM и т.д.).
    feature_names (array-like): Имена признаков после предобработки.
    model_name (str): Название модели для заголовка.
    top_n (int): Количество отображаемых признаков.
    """
    if hasattr(model, "feature_importances_"):
        # Для древовидных моделей (LightGBM, RandomForest и т.д.)
        importances = model.feature_importances_
        title = f'Топ-{top_n} важных признаков (feature importance)'
        xlabel = 'Важность признака'
        color = 'skyblue'
    elif hasattr(model, "coef_"):
        # Для линейных моделей (Ridge, Lasso, LinearRegression)
        importances = np.abs(model.coef_)
        title = f'Топ-{top_n} признаков по |коэффициенту|'
        xlabel = 'Абсолютное значение коэффициента'
        color = 'coral'
    else:
        raise AttributeError(f"Модель {type(model).__name__} не поддерживает ни feature_importances_, ни coef_")

    # Создаём DataFrame и сортируем
    importance_df = pd.DataFrame({
        'Признак': feature_names,
        'Важность': importances
    }).sort_values(by='Важность', ascending=False).head(top_n)

    # График
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['Важность'], color=color)
    plt.yticks(range(len(importance_df)), importance_df['Признак'])
    plt.xlabel(xlabel, fontsize=12)
    plt.title(f'{title}\n{model_name}', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Выводим таблицу
    print(f"Топ-{top_n} признаков для {model_name}:")
    print(importance_df.reset_index(drop=True))