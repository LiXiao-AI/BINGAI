"""
Test script to verify pipeline components work correctly
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from feature_engineering import TargetEncoder, TextFeatureExtractor


def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    
    # Sample training data
    train_data = {
        'Category': ['Algebra', 'Geometry', 'Algebra', 'Statistics', 'Geometry'] * 20,
        'Misconception': ['Linear equations', 'Area calculation', np.nan, 'Mean vs median', 'Angle measurement'] * 20,
        'QuestionText': ['Solve for x: 2x + 3 = 7'] * 100,
        'MC_Answer': ['x = 2', 'x = 3', 'x = 4', 'x = 5'] * 25,
        'StudentExplanation': [
            'I think x = 2 because 2*2 + 3 = 7',
            'Maybe x = 3?',
            'I divided both sides by 2',
            'Not sure about this one',
            'Used algebra to solve'
        ] * 20
    }
    
    # Sample test data
    test_data = {
        'row_id': range(1, 21),
        'QuestionText': ['Find the area of a rectangle'] * 20,
        'MC_Answer': ['12 cm²', '15 cm²', '18 cm²', '20 cm²'] * 5,
        'StudentExplanation': [
            'Length times width',
            'I multiplied the sides',
            'Area = l × w',
            'Not sure how to calculate'
        ] * 5
    }
    
    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def test_target_encoder():
    """Test target encoding functionality"""
    print("🧪 Testing Target Encoder...")
    
    train_df, _ = create_sample_data()
    encoder = TargetEncoder(rare_threshold=5)
    
    # Test target column creation
    train_df = encoder.create_target_column(train_df)
    assert 'Category:Misconception' in train_df.columns
    print("✅ Target column creation works")
    
    # Test rare class merging
    train_df = encoder.merge_rare_classes(train_df)
    print("✅ Rare class merging works")
    
    # Test encoding
    train_df, y = encoder.encode_target(train_df)
    assert len(y) == len(train_df)
    print("✅ Target encoding works")
    
    return encoder


def test_text_features():
    """Test text feature extraction"""
    print("\n🧪 Testing Text Feature Extractor...")
    
    train_df, test_df = create_sample_data()
    extractor = TextFeatureExtractor()
    
    # Test feature addition
    train_df = extractor.add_text_features(train_df)
    test_df = extractor.add_text_features(test_df)
    
    feature_names = extractor.get_feature_names()
    
    # Check if all features are present
    for feature in feature_names:
        assert feature in train_df.columns, f"Feature {feature} missing from train_df"
        assert feature in test_df.columns, f"Feature {feature} missing from test_df"
    
    print(f"✅ All {len(feature_names)} text features extracted successfully")
    print(f"Features: {feature_names[:5]}...")  # Show first 5 features
    
    return train_df, test_df


def test_config():
    """Test configuration"""
    print("\n🧪 Testing Configuration...")
    
    config = Config()
    config.update_paths_for_local()
    
    assert hasattr(config, 'TRAIN_PATH')
    assert hasattr(config, 'XGBOOST_PARAMS')
    assert hasattr(config, 'CV_FOLDS')
    
    print("✅ Configuration loaded successfully")
    print(f"CV Folds: {config.CV_FOLDS}")
    print(f"SVD Components: {config.SVD_COMPONENTS}")
    
    return config


def main():
    """Run all tests"""
    print("🚀 Starting Pipeline Component Tests")
    print("=" * 50)
    
    try:
        # Test configuration
        config = test_config()
        
        # Test target encoder
        encoder = test_target_encoder()
        
        # Test text features
        train_df, test_df = test_text_features()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("✅ Pipeline components are working correctly")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()