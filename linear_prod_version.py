import pandas as pd
from pathlib import Path
import numpy as np
from data_cleaning import clean_data
from data_preparation import data_preparation

INPUT_TEST_PATH = Path('data/raw/test.csv')
OUTPUT_PATH = Path('data/test_with_predicted_revenue/')

if __name__ == "__main__":
    test_df = pd.read_csv(INPUT_TEST_PATH)

    test_cleaned_df = clean_data(df=test_df)
    test_prepared_df = data_preparation(df=test_cleaned_df)

    # Check for missing values
    if test_prepared_df.isna().sum().any():
        print(test_prepared_df.isna().sum())
        raise ValueError("There is some NaN values in dataframe")

    # Apply linear baseline from EDA
    y_pred = np.exp(0.426147 + 0.993024 * test_prepared_df['log_rev_24h'])

    test_prepared_df = test_prepared_df.assign(revenue_30d_total=y_pred)

    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    test_prepared_df.to_csv(OUTPUT_PATH / 'linear_submission.csv', index=False)
