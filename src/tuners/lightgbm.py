from src.tuners.base import BoostTuner
import pandas as pd
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json


class LGBMBoostTuner(BoostTuner):

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
        super().__init__(X_valid=X_valid, y_valid=y_valid)
        self.X_train = X_train
        self.y_train = y_train
        self.regressor = None

    def _get_optuna_objective(self) -> callable:
        def objective_lgbm(trial):
            # Define the hyperparameters to be optimized
            params = {
                'objective': 'regression',
                'metric': 'custom',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 2, 100),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }

            # Train and evaluate XGBoost model with given hyperparameters
            model = LGBMRegressor(**params)
            model.fit(self.X_train, self.y_train, eval_metric=[self.eval_metric])
            y_pred = model.predict(self.X_valid)
            rmse = mean_squared_error(self.y_valid, y_pred, squared=False)

            return self.eval_metric(true_values=self.y_valid, predicted_values=y_pred)

        return objective_lgbm

    def tune_boost(self, n_trials):
        # Run Optuna to search for optimal hyperparameters
        objective = self._get_optuna_objective()
        study = optuna.create_study(direction='minimize')
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(objective, n_trials=n_trials)

        self.study = study

        # Train CatBoost model with optimal hyperparameters
        best_params = study.best_params
        self.regressor = LGBMRegressor(**best_params)
        self.regressor.fit(self.X_train, self.y_train, verbose=False)

        # Evaluate model on test set
        y_pred = self.regressor.predict(self.X_valid)
        rmse = mean_squared_error(self.y_valid, y_pred, squared=False)
        print("RMSE on validation set: {:.2f}".format(rmse))

    @staticmethod
    def load_regressor(input_folder: Path) -> LGBMRegressor:
        with open(str(input_folder / 'lgbmboost.json'), 'r') as j:
            best_params = json.loads(j.read())
        return LGBMRegressor(**best_params)
