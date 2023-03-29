from boosting_finetuning import main
from pathlib import Path
import random

INPUT_PATH = Path('data/prepared/')
OPTUNA_N_TRIALS = 100
OUTPUT_MODELS_FOLDER = Path('data/models/prepared_based/')
random.seed(42)

if __name__ == '__main__':
    main(input_path=INPUT_PATH, optuna_n_trials=OPTUNA_N_TRIALS, output_models_folder=OUTPUT_MODELS_FOLDER,
         finetune_catboost=False, finetune_xgboost=False, finetune_lgbmboost=False)
