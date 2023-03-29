import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from data_cleaning import clean_data
from data_preparation import data_preparation
from src.tuners.lightgbm import LGBMBoostTuner
import random

INPUT_PATH = Path('data/raw/train.csv')
INPUT_TEST_PATH = Path('data/raw/test.csv')
INPUT_MODEL_FOLDER = Path('data/models/prepared_based/')
OUTPUT_PATH = Path('data/test_with_predicted_revenue/')


if __name__ == "__main__":
    random.seed(42)
    df = pd.read_csv(INPUT_PATH)

    cleaned_df = clean_data(df=df)
    prepared_df = data_preparation(df=cleaned_df)

    # Check for missing values
    if prepared_df.isna().sum().any():
        print(prepared_df.isna().sum())
        raise ValueError("There is some NaN values in dataframe")

    # Split the data into features and target variable
    X = prepared_df.drop('revenue_30d_total', axis=1)
    y = prepared_df['revenue_30d_total']

    lbl = LabelEncoder()
    X['install_date'] = lbl.fit_transform(X['install_date'].astype(str))
    X['language'] = lbl.fit_transform(X['language'].astype(str))

    model = LGBMBoostTuner.load_regressor(input_folder=INPUT_MODEL_FOLDER)
    model.fit(X, y, verbose=True)

    test_df = pd.read_csv(INPUT_TEST_PATH)

    test_cleaned_df = clean_data(df=test_df)
    test_prepared_df = data_preparation(df=test_cleaned_df)

    X_test = test_prepared_df.drop('revenue_30d_total', axis=1)
    y_test = test_prepared_df['revenue_30d_total']

    X_test['install_date'] = lbl.fit_transform(X_test['install_date'].astype(str))
    X_test['language'] = lbl.fit_transform(X_test['language'].astype(str))

    # Check for missing values
    if test_prepared_df.isna().sum().any():
        print(test_prepared_df.isna().sum())
        raise ValueError("There is some NaN values in dataframe")

    y_pred = model.predict(X_test)

    test_prepared_df = test_prepared_df.assign(revenue_30d_total=y_pred)

    # Split the data into training and testing sets

    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    test_prepared_df.to_csv(OUTPUT_PATH / 'submission.csv', index=False)

