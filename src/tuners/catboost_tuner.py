from src.tuners.base import BoostTuner
import pandas as pd
from catboost import CatBoostRegressor, Pool
import optuna
from sklearn.metrics import mean_squared_error
import numpy as np
from pathlib import Path
import json


class CatBoostTuner(BoostTuner):

    def __init__(self, train_pool: Pool, X_valid: pd.DataFrame, y_valid: pd.Series):
        super().__init__(X_valid=X_valid, y_valid=y_valid)
        self.train_pool = train_pool
        self.regressor = None

    def _get_optuna_objective(self) -> callable:
        def objective_cat(trial):
            params = {
                'loss_function': 'RMSE',
                'iterations': trial.suggest_int('iterations', 100, 1000, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'depth': trial.suggest_int('depth', 3, 16),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 100),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 1)
            }

            # Train and evaluate CatBoost model with given hyperparameters
            model = CatBoostRegressor(**params)
            model.fit(self.train_pool, verbose=False)
            y_pred = model.predict(self.X_valid)
            rmse = mean_squared_error(self.y_valid, y_pred, squared=False)

            return self.eval_metric(self.y_valid, y_pred)

        return objective_cat

    def load_best_params(self, input_folder: Path) -> dict:
        return json.loads(str(input_folder / 'catboost.json'))

    def tune_boost(self, n_trials):
        # Run Optuna to search for optimal hyperparameters
        objective = self._get_optuna_objective()
        study = optuna.create_study(direction='minimize')
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(objective, n_trials=n_trials)

        self.study = study

        # Train CatBoost model with optimal hyperparameters
        best_params = study.best_params
        self.regressor = CatBoostRegressor(**best_params)
        self.regressor.fit(self.train_pool, verbose=False)

        # Evaluate model on test set
        y_pred = self.regressor.predict(self.X_valid)
        rmse = mean_squared_error(self.y_valid, y_pred, squared=False)
        print("RMSE on validation set: {:.2f}".format(rmse))
