import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from database import get_engine
from preprocessing import prepare_data
from models import train_and_evaluate
from visualization import plot_regression_analysis, plot_feature_importance, plot_price_distribution

def load_data_from_db(engine):
    """Загрузка данных напрямую из PostgreSQL"""
    print("Шаг 1: Загрузка данных из базы данных")
    try:
        train_df = pd.read_sql("SELECT * FROM train_data", engine)
        test_df = pd.read_sql("SELECT * FROM test_data", engine)
        
        if train_df.empty or test_df.empty:
            return print("Одна из таблиц в БД пуста!")
            
        return train_df, test_df
    except Exception as e:
        print(f"Ошибка при чтении из БД: {e}")

def main():
    engine = get_engine()
    if not engine:
        print("Не удалось подключиться к БД. Завершение работы.")
        return

    train_df, test_df = load_data_from_db(engine)
    
    plot_price_distribution(train_df)
    
    print("Шаг 2: Предобработка данных")
    X_train, X_val, y_train, y_val, preprocessor = prepare_data(train_df)
    print(f"Размер обучающей выборки: {X_train.shape}")
    
    print("Шаг 3: Обучение ансамбля моделей (Stacking)")
    results_df, best_model = train_and_evaluate(X_train, X_val, y_train, y_val)
    
    print("\nМетрики на валидации:")
    print(results_df.to_string(index=False))
    
    print("Шаг 4: Визуализация и сохранение артефактов")
    pred_val_log = best_model.predict(X_val)
    plot_regression_analysis(y_val, pred_val_log, model_name="Stacked Ensemble")
    
    feature_names = preprocessor.get_feature_names_out()
    plot_feature_importance(best_model, feature_names, model_name="Stacking Model")

    print("Шаг 5: Формирование финального предсказания")
    
    cols_to_drop = ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Id']
    X_test = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns], errors='ignore')
    
    X_test_raw = preprocessor.transform(X_test)
    X_test_prep = pd.DataFrame(X_test_raw, columns=preprocessor.get_feature_names_out())
    
    final_preds_log = best_model.predict(X_test_prep)
    final_preds_price = np.expm1(np.maximum(final_preds_log, 0))
    
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': final_preds_price
    })
    
    submission.to_csv('submission_final.csv', index=False)
    print("Файл 'submission_final.csv' успешно создан!")

if __name__ == "__main__":
    main()