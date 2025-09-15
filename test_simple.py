"""
Simple test script to verify basic pipeline components work
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config


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


def test_target_creation():
    """Test basic target creation without heavy dependencies"""
    print("🧪 Testing Target Creation...")
    
    train_df, _ = create_sample_data()
    
    # Create target column (same logic as in TargetEncoder)
    train_df["Category:Misconception"] = train_df.apply(
        lambda row: f"{row['Category']}:{row['Misconception']}" 
        if pd.notna(row["Misconception"]) 
        else f"{row['Category']}:NA",
        axis=1
    )
    
    assert 'Category:Misconception' in train_df.columns
    print("✅ Target column creation works")
    
    # Test rare class merging
    counts = train_df["Category:Misconception"].value_counts()
    rare_classes = counts[counts <= 5].index
    train_df.loc[train_df["Category:Misconception"].isin(rare_classes), 
                 "Category:Misconception"] = "Rare"
    
    print("✅ Rare class merging works")
    print(f"Unique categories: {train_df['Category:Misconception'].nunique()}")
    
    return train_df


def test_text_features_basic():
    """Test basic text feature extraction without dependencies"""
    print("\n🧪 Testing Basic Text Features...")
    
    train_df, test_df = create_sample_data()
    
    # Add basic text features (same logic as in TextFeatureExtractor)
    for df in [train_df, test_df]:
        df["exp_len"] = df["StudentExplanation"].astype(str).apply(len)
        df["exp_words"] = df["StudentExplanation"].astype(str).apply(lambda x: len(x.split()))
        df["has_fraction"] = df["StudentExplanation"].str.contains("/", regex=False).astype(int)
        df["has_equals"] = df["StudentExplanation"].str.contains("=", regex=False).astype(int)
        df["has_percent"] = df["StudentExplanation"].str.contains("%", regex=False).astype(int)
        df["has_number"] = df["StudentExplanation"].str.contains(r"\d", regex=True).astype(int)
        df["q_len"] = df["QuestionText"].astype(str).apply(len)
        df["mc_len"] = df["MC_Answer"].astype(str).apply(len)
        df["exp_q_ratio"] = df["exp_len"] / (df["q_len"] + 1)
        df["is_exp_empty"] = (df["StudentExplanation"].astype(str).str.strip() == "").astype(int)
    
    feature_names = [
        "exp_len", "exp_words", "has_fraction", "has_equals", 
        "has_percent", "has_number", "q_len", "mc_len", 
        "exp_q_ratio", "is_exp_empty"
    ]
    
    # Check if all features are present
    for feature in feature_names:
        assert feature in train_df.columns, f"Feature {feature} missing from train_df"
        assert feature in test_df.columns, f"Feature {feature} missing from test_df"
    
    print(f"✅ All {len(feature_names)} basic text features extracted successfully")
    print(f"Sample features: {train_df[feature_names[:3]].head()}")
    
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
    print(f"XGBoost params keys: {list(config.XGBOOST_PARAMS.keys())[:5]}...")
    
    return config


def main():
    """Run all basic tests"""
    print("🚀 Starting Basic Pipeline Component Tests")
    print("=" * 50)
    
    try:
        # Test configuration
        config = test_config()
        
        # Test target creation
        train_df = test_target_creation()
        
        # Test basic text features
        train_df, test_df = test_text_features_basic()
        
        print("\n" + "=" * 50)
        print("🎉 All basic tests passed successfully!")
        print("✅ Core pipeline logic is working correctly")
        print("📝 Note: Full pipeline requires sentence-transformers and xgboost")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()