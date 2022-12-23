import pandas as pd
import numpy as np

class DataPreprocessing():

    def transform(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Конвертація фаренгейтів на цельсій
        df['value'] = (df['value'] - 32) * 5 / 9
        df['hours'] = df['timestamp'].dt.hour
        df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
        # День тижня аби розріняти робочі дні від вихідних.
        df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
        df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
        # Аби простіше було вивести на графіфку час
        df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)
        # Створення 4-ьох категорій (Вихідний/робочий день  & ніч/день)
        df['categories'] = df['WeekDay'] * 2 + df['daylight']

        return df