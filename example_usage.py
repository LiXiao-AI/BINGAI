"""
Example Usage Script for Math Misconception Classification Pipeline
Demonstrates how to use individual components of the pipeline
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from feature_engineering_simple import TargetEncoder, TextFeatureExtractor


def example_target_encoding():
    """Example of target encoding process"""
    print("🎯 Target Encoding Example")
    print("-" * 40)
    
    # Sample data
    data = {
        'Category': ['Algebra', 'Geometry', 'Algebra', 'Statistics'],
        'Misconception': ['Linear equations', 'Area calculation', np.nan, 'Mean vs median'],
        'QuestionText': ['Solve for x', 'Find area', 'Simplify', 'Calculate mean'],
        'MC_Answer': ['x = 2', '12 cm²', '3x', '5.5'],
        'StudentExplanation': ['I solved it', 'Length times width', 'Combined terms', 'Added and divided']
    }
    
    df = pd.DataFrame(data)
    print("Original data:")
    print(df[['Category', 'Misconception']].head())
    
    # Initialize encoder
    encoder = TargetEncoder(rare_threshold=1)  # Low threshold for demo
    
    # Create target column
    df = encoder.create_target_column(df)
    print(f"\nAfter creating target column:")
    print(df['Category:Misconception'].tolist())
    
    # Merge rare classes and encode
    df = encoder.merge_rare_classes(df)
    df, y = encoder.encode_target(df)
    
    print(f"\nFinal encoded targets: {y}")
    print(f"Label classes: {encoder.label_encoder.classes_}")


def example_text_features():
    """Example of text feature extraction"""
    print("\n\n📝 Text Feature Extraction Example")
    print("-" * 40)
    
    # Sample data with mathematical content
    data = {
        'QuestionText': [
            'Solve for x: 2x + 3 = 7',
            'What is 25% of 80?',
            'Find the area of a rectangle with length 5 and width 3'
        ],
        'MC_Answer': ['x = 2', '20', '15 square units'],
        'StudentExplanation': [
            'I subtracted 3 from both sides, then divided by 2',
            'I think 25% = 1/4, so 80/4 = 20',
            'Area = length × width = 5 × 3 = 15'
        ]
    }
    
    df = pd.DataFrame(data)
    print("Original explanations:")
    for i, exp in enumerate(df['StudentExplanation']):
        print(f"{i+1}. {exp}")
    
    # Extract features
    extractor = TextFeatureExtractor()
    df = extractor.add_text_features(df)
    
    # Show some key features
    key_features = ['exp_len', 'exp_words', 'has_fraction', 'has_equals', 'has_number']
    print(f"\nExtracted features:")
    print(df[key_features])
    
    # Show all available features
    all_features = extractor.get_feature_names()
    print(f"\nAll {len(all_features)} available features:")
    for i, feature in enumerate(all_features):
        if i % 5 == 0:
            print()
        print(f"{feature:15}", end=" ")
    print()


def example_configuration():
    """Example of configuration usage"""
    print("\n\n⚙️ Configuration Example")
    print("-" * 40)
    
    config = Config()
    
    print("Default Kaggle paths:")
    print(f"Train: {config.TRAIN_PATH}")
    print(f"Test: {config.TEST_PATH}")
    print(f"Model: {config.MODEL_PATH}")
    
    print(f"\nXGBoost parameters:")
    for key, value in list(config.XGBOOST_PARAMS.items())[:5]:
        print(f"  {key}: {value}")
    print("  ...")
    
    print(f"\nCross-validation settings:")
    print(f"  Folds: {config.CV_FOLDS}")
    print(f"  Random state: {config.CV_RANDOM_STATE}")
    print(f"  Early stopping: {config.EARLY_STOPPING_ROUNDS}")
    
    # Switch to local paths
    config.update_paths_for_local()
    print(f"\nLocal development paths:")
    print(f"Train: {config.TRAIN_PATH}")
    print(f"Test: {config.TEST_PATH}")


def example_feature_analysis():
    """Example of analyzing extracted features"""
    print("\n\n📊 Feature Analysis Example")
    print("-" * 40)
    
    # Create diverse sample data
    explanations = [
        "I solved 2x + 3 = 7 by subtracting 3 from both sides",
        "Maybe it's 25%? I'm not sure...",
        "Area = length × width = 5 × 3 = 15 cm²",
        "I think the answer is (a) but I could be wrong",
        "Used the formula: A = πr² where r = 3",
        "",  # Empty explanation
        "50/100 = 0.5 = 50%",
        "What does this question mean???"
    ]
    
    data = {
        'QuestionText': ['Math problem'] * len(explanations),
        'MC_Answer': ['Answer'] * len(explanations),
        'StudentExplanation': explanations
    }
    
    df = pd.DataFrame(data)
    extractor = TextFeatureExtractor()
    df = extractor.add_text_features(df)
    
    # Analyze feature distributions
    analysis_features = [
        'exp_len', 'exp_words', 'has_fraction', 'has_equals', 
        'has_number', 'uncertainty_words', 'is_exp_empty'
    ]
    
    print("Feature analysis across different explanation types:")
    print(df[analysis_features].describe())
    
    # Show which explanations have specific characteristics
    print(f"\nExplanations with fractions: {df[df['has_fraction'] == 1].index.tolist()}")
    print(f"Explanations with uncertainty: {df[df['uncertainty_words'] > 0].index.tolist()}")
    print(f"Empty explanations: {df[df['is_exp_empty'] == 1].index.tolist()}")


def main():
    """Run all examples"""
    print("🚀 Math Misconception Pipeline - Usage Examples")
    print("=" * 60)
    
    try:
        example_target_encoding()
        example_text_features()
        example_configuration()
        example_feature_analysis()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("🔗 These components can be combined in the main pipeline")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()