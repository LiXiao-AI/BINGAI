"""
Configuration file for the Math Misconception Classification Pipeline
"""

import os

class Config:
    # Data paths
    TRAIN_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
    TEST_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
    MODEL_PATH = "/kaggle/input/bge_embeddings/pytorch/default/1/finetuned_bge_embeddings_v5_small_v1.5"
    
    # Output paths
    OUTPUT_DIR = 'outputs'
    SUBMISSION_FILE = 'submission.csv'
    
    # Model parameters
    EMBEDDING_BATCH_SIZE = 64
    SVD_COMPONENTS = 128
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'n_estimators': 3000,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.5,
        'reg_lambda': 1.5,
        'min_child_weight': 2,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'random_state': 42
    }
    
    # Cross-validation parameters
    CV_FOLDS = 5
    CV_RANDOM_STATE = 42
    EARLY_STOPPING_ROUNDS = 50
    
    # Feature engineering parameters
    RARE_CLASS_THRESHOLD = 5
    
    # Random seeds
    RANDOM_STATE = 42
    
    @classmethod
    def update_paths_for_local(cls):
        """Update paths for local development"""
        cls.TRAIN_PATH = 'data/train.csv'
        cls.TEST_PATH = 'data/test.csv'
        cls.MODEL_PATH = 'models/sentence_transformer_model'