from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

def get_engine():
    DB_HOST = 'localhost'
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    return engine

def load_data():
    from pandas import read_sql
    engine = get_engine()
    train_df = read_sql("SELECT * FROM train", engine)
    test_df = read_sql("SELECT * FROM test", engine)
    return train_df, test_df