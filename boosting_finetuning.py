import numpy as np
import pandas as pd
from catboost import Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pathlib import Path
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from src.tuners.catboost_tuner import CatBoostTuner
from src.tuners.xgboost_tuner import XGBoostTuner
from src.tuners.lightgbm import LGBMBoostTuner
import random

INPUT_PATH = Path('data/cleaned/')
OPTUNA_N_TRIALS = 40
OUTPUT_MODELS_FOLDER = Path('data/models/cleaned_based/')


def defined_metric(true_values: pd.Series, predicted_values: np.ndarray) -> tuple[float, float]:
    metric_values = (pd.Series(predicted_values) - true_values) / true_values
    mae = mean_absolute_error(true_values, pd.Series(predicted_values))
    return (metric_values.replace([np.inf, -np.inf], np.nan).mean(), mae)


def apply_boosting(regressor, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
    return defined_metric(true_values=y_test, predicted_values=regressor.predict(x_test))


def main(input_path: Path, optuna_n_trials: int, output_models_folder: Path,
         finetune_catboost: bool = True, finetune_xgboost: bool = True, finetune_lgbmboost: bool = True):
    X_train = pd.read_csv(input_path / 'X_train.csv')
    X_test = pd.read_csv(input_path / 'X_test.csv')
    y_train = pd.read_csv(input_path / 'y_train.csv')
    y_test = pd.read_csv(input_path / 'y_test.csv')

    categorical_features_indices = ['install_date', 'language', 'api_level', 'cpu_cores']

    lbl = preprocessing.LabelEncoder()
    if finetune_xgboost or finetune_lgbmboost:
        X_train['install_date'] = lbl.fit_transform(X_train['install_date'].astype(str))
        X_train['language'] = lbl.fit_transform(X_train['language'].astype(str))
    X_test['install_date'] = lbl.fit_transform(X_test['install_date'].astype(str))
    X_test['language'] = lbl.fit_transform(X_test['language'].astype(str))

    if finetune_catboost:
        train_pool = Pool(data=X_train,
                          label=y_train,
                          cat_features=categorical_features_indices)

        test_pool = Pool(data=X_test,
                         label=y_test,
                         cat_features=categorical_features_indices)

    if finetune_lgbmboost:
        lgbm_tuner = LGBMBoostTuner(X_train=X_train,
                                    y_train=y_train['revenue_30d_total'],
                                    X_valid=X_test,
                                    y_valid=y_test['revenue_30d_total'])
        lgbm_tuner.tune_boost(optuna_n_trials)
        lgbm_tuner.save_best_params(output_path=output_models_folder / 'lgbmboost.json')
        lgbm_model_err = apply_boosting(regressor=lgbm_tuner.regressor,
                                        x_test=X_test,
                                        y_test=y_test['revenue_30d_total'])
        print(f'LightGBM baseline: \n'
              f'target_metric: {lgbm_model_err[0]} \n'
              f'mae: {lgbm_model_err[1]}')

    if finetune_catboost:
        cat_tuner = CatBoostTuner(train_pool=train_pool,
                                  X_valid=X_test,
                                  y_valid=y_test['revenue_30d_total'])
        cat_tuner.tune_boost(optuna_n_trials)
        cat_tuner.save_best_params(output_path=output_models_folder / 'catboost.json')

        catboost_model_err = apply_boosting(regressor=cat_tuner.regressor,
                                            x_test=X_test,
                                            y_test=y_test['revenue_30d_total'])

        print(f'CatBoost baseline:\n'
              f'target_metric: {catboost_model_err[0]} \n'
              f'mae: {catboost_model_err[1]}')

    if finetune_xgboost:
        xgb_tuner = XGBoostTuner(X_train=X_train,
                                 y_train=y_train['revenue_30d_total'],
                                 X_valid=X_test,
                                 y_valid=y_test['revenue_30d_total'])
        xgb_tuner.tune_boost(optuna_n_trials)
        xgb_tuner.save_best_params(output_path=output_models_folder / 'xgboost.json')

        xgb_model_err = apply_boosting(regressor=xgb_tuner.regressor,
                                       x_test=X_test,
                                       y_test=y_test['revenue_30d_total'])

        print(f'XGB baseline: \n'
              f'target_metric: {xgb_model_err[0]} \n'
              f'mae: {xgb_model_err[1]}')

    y_pred_log_baseline = np.exp(0.426147 + 0.993024 * X_test['log_rev_24h'])

    print('Linear baseline Evaluation metric: ', LGBMBoostTuner.eval_metric(true_values=y_test['revenue_30d_total'],
                                                                            predicted_values=y_pred_log_baseline))


if __name__ == '__main__':
    random.seed(42)
    main(input_path=INPUT_PATH, optuna_n_trials=OPTUNA_N_TRIALS, output_models_folder=OUTPUT_MODELS_FOLDER,
         finetune_catboost=True, finetune_xgboost=False, finetune_lgbmboost=False)
