"""
Main Pipeline for Math Misconception Classification
Enhanced version of the original Kaggle competition solution
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from feature_engineering import FeaturePipeline
from model_training import ModelTrainer, PredictionGenerator


def main():
    """Main pipeline execution"""
    print("🚀 Starting Math Misconception Classification Pipeline")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Check if running locally and update paths if needed
    if not os.path.exists(config.TRAIN_PATH):
        print("⚠️  Kaggle paths not found, switching to local paths")
        config.update_paths_for_local()
    
    # Verify data files exist
    if not os.path.exists(config.TRAIN_PATH):
        print(f"❌ Training data not found at {config.TRAIN_PATH}")
        print("Please ensure data files are in the correct location")
        return
    
    if not os.path.exists(config.TEST_PATH):
        print(f"❌ Test data not found at {config.TEST_PATH}")
        print("Please ensure data files are in the correct location")
        return
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # -------------------- Step 1: Load Data --------------------
    print("\n📂 Loading data...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # -------------------- Step 2: Feature Engineering --------------------
    print("\n🔧 Starting feature engineering...")
    feature_pipeline = FeaturePipeline(config)
    
    # Process training data
    X_train, y_train, label_encoder = feature_pipeline.process_training_data(train_df)
    
    # Process test data
    X_test = feature_pipeline.process_test_data(test_df)
    
    print(f"Final feature dimensions - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # -------------------- Step 3: Model Training --------------------
    print("\n🤖 Starting model training...")
    trainer = ModelTrainer(config)
    
    # Cross-validation training
    models, cv_score = trainer.cross_validate_train(X_train, y_train)
    
    # Save models
    trainer.save_models(os.path.join(config.OUTPUT_DIR, 'models'))
    
    # -------------------- Step 4: Prediction --------------------
    print("\n🔮 Generating predictions...")
    test_predictions = trainer.predict_test(X_test)
    
    # Create submission
    submission_path = os.path.join(config.OUTPUT_DIR, config.SUBMISSION_FILE)
    submission = PredictionGenerator.create_submission(
        test_df, test_predictions, label_encoder, submission_path
    )
    
    # Analyze predictions
    pred_analysis = PredictionGenerator.analyze_predictions(
        test_predictions, label_encoder
    )
    
    # Save prediction analysis
    analysis_path = os.path.join(config.OUTPUT_DIR, 'prediction_analysis.csv')
    pred_analysis.to_csv(analysis_path, index=False)
    
    # -------------------- Step 5: Summary --------------------
    print("\n" + "=" * 60)
    print("🎉 Pipeline completed successfully!")
    print(f"📊 Cross-validation score: {cv_score:.4f}")
    print(f"📁 Outputs saved to: {config.OUTPUT_DIR}/")
    print(f"📄 Submission file: {submission_path}")
    print(f"📈 Analysis file: {analysis_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()