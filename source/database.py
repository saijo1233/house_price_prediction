import os
import pandas as pd
from sqlalchemy import create_engine, Engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()

def get_engine() -> Optional[Engine]:
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5433')
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    
    url = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    
    try:
        engine = create_engine(url, echo=False)
        with engine.connect() as connection:
            print(f"Подключено к базе '{DB_NAME}' на порту {DB_PORT}")
            return engine
    except SQLAlchemyError as e:
        print(f"Ошибка подключения: {e}")
        return None

def load_data():
    base_path = 'data' 
    train_path = os.path.join(base_path, 'train.csv')
    test_path = os.path.join(base_path, 'test.csv')
    
    def read_csv_with_fallback(path: str) -> pd.DataFrame:
        for encoding in ['utf-8', 'cp1251', 'utf-8-sig']:
            try:
                return pd.read_csv(path, encoding=encoding, na_values=['NA'])
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"Файл не найден: {path}")
                raise
        raise ValueError(f"Не удалось прочитать файл {path}.")

    train_df = read_csv_with_fallback(train_path)
    test_df = read_csv_with_fallback(test_path)
    return train_df, test_df

def save_to_db(engine: Engine, train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        print("Загрузка train.csv...")
        train_df.to_sql('train', engine, if_exists='replace', index=False)
        
        print("Загрузка test.csv...")
        test_df.to_sql('test', engine, if_exists='replace', index=False)
        
        print("Данные успешно загружены в Docker!")
    except Exception as e:
        print(f"Ошибка при сохранении в базу: {e}")

if __name__ == "__main__":
    engine = get_engine()
    
    if engine:
        train_df, test_df = load_data()
        
        save_to_db(engine, train_df, test_df)