from sqlalchemy import create_engine
from pandas import read_sql
import pandas as pd
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
    engine = get_engine()
    try:
        train_df = pd.read_csv('C:/Users/User/Desktop/project/data/train.csv', encoding='cp1251')
        test_df = pd.read_csv('C:/Users/User/Desktop/project/data/test.csv', encoding='cp1251')
    except UnicodeDecodeError:
        train_df = pd.read_csv('C:/Users/User/Desktop/project/data/train.csv', encoding='utf-8-sig')
        test_df = pd.read_csv('C:/Users/User/Desktop/project/data/test.csv', encoding='utf-8-sig')
    return train_df, test_df