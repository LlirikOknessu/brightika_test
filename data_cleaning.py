import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

INPUT_PATH = Path('data/raw/train.csv')
OUTPUT_PATH = Path('data/cleaned/')
EPSILON = 1e-9


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data of the dataframe: replace default values and NaN by mean,
    drop unnecessary columns, add another representation of revenues.
    :param df: Dataframe for preparation
    :return: prepared dataframe
    """
    df = df.drop(columns=['random_user_id', 'ua_network_name', 'country_code', 'user_agent'])
    fill_columns = ['conversion_duration', 'device_price', 'device_ram', 'cpu_cores', 'screen_inches_diagonal',
                    'screen_pixels_width', 'screen_pixels_height']
    fill_int = ['conversion_duration', 'device_price', 'device_ram', 'cpu_cores', 'screen_pixels_width',
                'screen_pixels_height']
    fill_float = ['screen_inches_diagonal']
    fill_cat = ['language', 'api_level']

    df[fill_columns] = df[fill_columns].replace(0, np.NaN)
    df[fill_int] = df[fill_int].fillna(df[fill_int].mean().astype(int))
    df[fill_float] = df[fill_float].fillna(df[fill_float].mean())
    df['language'] = df['language'].fillna(df['language'].value_counts().idxmin())
    df['api_level'] = df['api_level'].fillna(df['api_level'].value_counts().idxmin())
    df[['api_level', 'cpu_cores']] = df[['api_level', 'cpu_cores']].astype(int)
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    df = clean_data(df=df)

    # Check for missing values
    if df.isna().sum().any():
        print(df.isna().sum())
        raise ValueError("There is some NaN values in dataframe")

    # Split the data into features and target variable
    X = df.drop('revenue_30d_total', axis=1)
    y = df['revenue_30d_total']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    X_train.to_csv(OUTPUT_PATH / 'X_train.csv', index=False)
    X_test.to_csv(OUTPUT_PATH / 'X_test.csv', index=False)

    y_train.to_csv(OUTPUT_PATH / 'y_train.csv', index=False)
    y_test.to_csv(OUTPUT_PATH / 'y_test.csv', index=False)
