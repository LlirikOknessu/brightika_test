import pandas as pd
from optuna.study import Study
from pathlib import Path
import json
import numpy as np


class BoostTuner:
    def __init__(self, X_valid: pd.DataFrame, y_valid: pd.Series):
        self.regressor = None
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.study: Study = None

    def _get_optuna_objective(self) -> callable:
        pass

    def tune_boost(self, n_trials):
        pass

    def save_best_params(self, output_path: Path):
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(str(output_path), 'w') as f:
            json.dump(self.study.best_params, f)

    def load_best_params(self, input_folder: Path):
        pass

    @staticmethod
    def eval_metric(true_values, predicted_values) -> tuple[str, float, bool]:
        metric_values = (pd.Series(predicted_values) - true_values) / true_values
        eval_metric = metric_values.replace([np.inf, -np.inf], np.nan).mean()
        return eval_metric