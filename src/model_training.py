"""
Model Training Module for Math Misconception Classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from typing import Tuple, List
import os


class ModelTrainer:
    """Handles model training with cross-validation"""
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.val_scores = []
    
    def compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute balanced sample weights for class imbalance"""
        classes = np.unique(y)
        class_weights = compute_class_weight(
            class_weight="balanced", 
            classes=classes, 
            y=y
        )
        sample_weights = class_weights[y]
        return sample_weights
    
    def train_fold(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   sample_weights: np.ndarray, fold: int) -> xgb.XGBClassifier:
        """Train a single fold"""
        print(f"\n--- Training Fold {fold} ---")
        
        # Update num_class parameter
        params = self.config.XGBOOST_PARAMS.copy()
        params['num_class'] = len(np.unique(y_train))
        
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
            verbose=100
        )
        
        best_iter = model.best_iteration
        val_logloss = model.evals_result()["validation_0"]["mlogloss"][best_iter]
        self.val_scores.append(val_logloss)
        
        print(f"Fold {fold} best logloss: {val_logloss:.4f}")
        return model
    
    def cross_validate_train(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[xgb.XGBClassifier], float]:
        """Perform stratified k-fold cross-validation training"""
        print("🚀 Starting StratifiedKFold training...")
        
        # Compute sample weights
        sample_weights = self.compute_sample_weights(y)
        
        # Initialize cross-validation
        kf = StratifiedKFold(
            n_splits=self.config.CV_FOLDS, 
            shuffle=True, 
            random_state=self.config.CV_RANDOM_STATE
        )
        
        self.models = []
        self.val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = sample_weights[train_idx]
            
            model = self.train_fold(X_tr, y_tr, X_val, y_val, w_tr, fold)
            self.models.append(model)
        
        mean_cv_score = np.mean(self.val_scores)
        print(f"\n📊 CV logloss: {mean_cv_score:.4f} ± {np.std(self.val_scores):.4f}")
        
        return self.models, mean_cv_score
    
    def predict_test(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions using ensemble of trained models"""
        if not self.models:
            raise ValueError("No trained models found. Run cross_validate_train first.")
        
        print("🔮 Generating ensemble predictions...")
        test_pred_proba = np.zeros((X_test.shape[0], self.models[0].n_classes_))
        
        for model in self.models:
            test_pred_proba += model.predict_proba(X_test) / len(self.models)
        
        return test_pred_proba
    
    def save_models(self, output_dir: str):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            model_path = os.path.join(output_dir, f'xgb_fold_{i+1}.json')
            model.save_model(model_path)
            print(f"Model fold {i+1} saved to {model_path}")


class PredictionGenerator:
    """Handles final prediction generation and submission creation"""
    
    @staticmethod
    def create_submission(test_df: pd.DataFrame, predictions: np.ndarray, 
                         label_encoder, output_path: str) -> pd.DataFrame:
        """Create submission file from predictions"""
        test_preds = np.argmax(predictions, axis=1)
        test_labels = label_encoder.inverse_transform(test_preds)
        
        submission = pd.DataFrame({
            "row_id": test_df["row_id"],
            "Category:Misconception": test_labels
        })
        
        submission.to_csv(output_path, index=False)
        print(f"✅ Submission saved: {output_path}")
        
        return submission
    
    @staticmethod
    def analyze_predictions(predictions: np.ndarray, label_encoder) -> pd.DataFrame:
        """Analyze prediction distribution"""
        test_preds = np.argmax(predictions, axis=1)
        test_labels = label_encoder.inverse_transform(test_preds)
        
        pred_counts = pd.Series(test_labels).value_counts()
        pred_analysis = pd.DataFrame({
            'Category:Misconception': pred_counts.index,
            'Count': pred_counts.values,
            'Percentage': (pred_counts.values / len(test_labels) * 100).round(2)
        })
        
        print("\n📈 Prediction Distribution:")
        print(pred_analysis.head(10))
        
        return pred_analysis