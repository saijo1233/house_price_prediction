from database import load_data
from preprocessing import prepare_data
from models import train_and_evaluate
from visualization import plot_predictions, plot_feature_importance

def main():
    print("Загрузка данных...")
    train_df, _ = load_data()
    
    print("Предобработка...")
    X_train, X_val, y_train, y_val, preprocessor = prepare_data(train_df)
    
    print("Обучение моделей...")
    results_df, best_model = train_and_evaluate(X_train, X_val, y_train, y_val)
    print("\nРезультаты:")
    print(results_df)
    
    print("\nВизуализация...")
    pred_val = best_model.predict(X_val)
    plot_predictions(y_val, pred_val)
    
    feature_names = preprocessor.get_feature_names_out()
    plot_feature_importance(best_model, feature_names)
    
    print("\nГотово! Средняя ошибка ~${:.0f}".format(results_df['MAE ($)'].iloc[0]))

if __name__ == "__main__":
    main()