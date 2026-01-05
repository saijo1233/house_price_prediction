import numpy as np
import pandas as pd
from database import load_data
from preprocessing import prepare_data
from models import train_and_evaluate
from visualization import plot_regression_analysis, plot_feature_importance, plot_price_distribution

def main():
    print("\nШаг 1: Загрузка данных")
    train_df, test_df = load_data()
    
    plot_price_distribution(train_df)
    
    print("\nШаг 2: Предобработка данных")
    X_train, X_val, y_train, y_val, preprocessor = prepare_data(train_df)
    print(f"Размер обучающей выборки после обработки: {X_train.shape}")
    
    print("\nШаг 3: Обучение ансамбля моделей (Stacking)")
    results_df, best_model = train_and_evaluate(X_train, X_val, y_train, y_val)
    
    print("\nМетрики на валидационной выборке:")
    print(results_df.to_string(index=False))
    
    print("\n--- Шаг 4: Визуализация ---")
    pred_val_log = best_model.predict(X_val)
    
    plot_regression_analysis(y_val, pred_val_log, model_name="Stacked Ensemble")
    
    feature_names = preprocessor.get_feature_names_out()
    plot_feature_importance(best_model, feature_names, model_name="Stacking Model")
    
    print("\n--- Шаг 5: Формирование финального предсказания ---")

    cols_to_drop = ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    
    X_test = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns], errors='ignore')
    X_test = X_test.drop(columns=['Id'], errors='ignore')
    
    X_test_raw = preprocessor.transform(X_test)
    
    X_test_prep = pd.DataFrame(
        X_test_raw, 
        columns=preprocessor.get_feature_names_out()
    )
    
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