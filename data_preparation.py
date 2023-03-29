import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from data_cleaning import clean_data

INPUT_PATH = Path('data/raw/train.csv')
OUTPUT_PATH = Path('data/prepared/')
EPSILON = 1e-9


def data_preparation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data of the dataframe: replace default values and NaN by mean,
    drop unnecessary columns, add another representation of revenues.
    :param df: Dataframe for preparation
    :return: prepared dataframe
    """
    result = df.groupby(["language"])['revenue_30d_total'].mean().reset_index().sort_values('revenue_30d_total')
    lang_min = result.loc[result['revenue_30d_total'] < 0.1]['language'].tolist()
    lang_pre_mid = result.loc[(result['revenue_30d_total'] > 0.1) & (result['revenue_30d_total'] < 0.2)][
        'language'].tolist()
    lang_mid = result.loc[(result['revenue_30d_total'] > 0.2) & (result['revenue_30d_total'] < 0.3)][
        'language'].tolist()
    lang_pre_max = result.loc[(result['revenue_30d_total'] > 0.3) & (result['revenue_30d_total'] < 0.5)][
        'language'].tolist()
    lang_max = result.loc[result['revenue_30d_total'] > 0.5][
        'language'].tolist()
    # TODO Implement encoding language without warnings
    df['language'][df['language'].isin(lang_min)] = 0
    df['language'][df['language'].isin(lang_pre_mid)] = 1
    df['language'][df['language'].isin(lang_mid)] = 2
    df['language'][df['language'].isin(lang_pre_max)] = 3
    df['language'][df['language'].isin(lang_max)] = 4
    df = df.assign(square_area=df['screen_pixels_width'] * df['screen_pixels_height'])
    df = df.assign(log_rev_24h=np.log(df['revenue_24h_total'] + EPSILON))
    df = df.assign(ratio_revenue_24h_banner=(df['revenue_24h_banner'] / df['revenue_24h_total']))
    df = df.assign(ratio_revenue_24h_inters=(df['revenue_24h_inters'] / df['revenue_24h_total']))
    df = df.assign(ratio_revenue_24h_rewards=(df['revenue_24h_rewards'] / df['revenue_24h_total']))
    df = df.assign(mean_banner=df['revenue_24h_banner'] / df['ad_views_24h_banner'])
    df = df.assign(mean_inters=df['revenue_24h_inters'] / df['ad_views_24h_inters'])
    df = df.assign(mean_rewards=df['revenue_24h_rewards'] / df['ad_views_24h_rewards'])
    df = df.assign(mean_total_by_session=df['revenue_24h_total'] / df['sessions_24h'])
    fill_float = ['ratio_revenue_24h_banner', 'ratio_revenue_24h_inters', 'ratio_revenue_24h_rewards',
                  'mean_banner', 'mean_inters', 'mean_rewards', 'mean_total_by_session']
    df[fill_float] = df[fill_float].fillna(df[fill_float].mean())
    return df


if __name__ == "__main__":
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

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    X_train.to_csv(OUTPUT_PATH / 'X_train.csv', index=False)
    X_test.to_csv(OUTPUT_PATH / 'X_test.csv', index=False)

    y_train.to_csv(OUTPUT_PATH / 'y_train.csv', index=False)
    y_test.to_csv(OUTPUT_PATH / 'y_test.csv', index=False)
