from src.tuners.base import BoostTuner
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
from pathlib import Path
import json


class XGBoostTuner(BoostTuner):

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
        super().__init__(X_valid=X_valid, y_valid=y_valid)
        self.X_train = X_train
        self.y_train = y_train
        self.regressor = None

    def _get_optuna_objective(self) -> callable:
        def objective_xgb(trial):
            params = {
                'objective': 'reg:squarederror',
                'booster': 'gbtree',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.5, 1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5)
            }

            # Train and evaluate XGBoost model with given hyperparameters
            model = XGBRegressor(**params, eval_metric=self.eval_metric(self.y_valid, y_pred))
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_valid)
            rmse = mean_squared_error(self.y_valid, y_pred, squared=False)

            return self.eval_metric(self.y_valid, y_pred)

        return objective_xgb

    def load_best_params(self, input_folder: Path) -> dict:
        return json.loads(str(input_folder / 'xgboost.json'))

    def tune_boost(self, n_trials):
        # Run Optuna to search for optimal hyperparameters
        objective = self._get_optuna_objective()
        study = optuna.create_study(direction='minimize')
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(objective, n_trials=n_trials)

        self.study = study

        # Train CatBoost model with optimal hyperparameters
        best_params = study.best_params
        self.regressor = XGBRegressor(**best_params)
        self.regressor.fit(self.X_train, self.y_train, verbose=False)

        # Evaluate model on test set
        y_pred = self.regressor.predict(self.X_valid)
        rmse = mean_squared_error(self.y_valid, y_pred, squared=False)
        print("RMSE on validation set: {:.2f}".format(rmse))
