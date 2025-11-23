"""
Simplified Feature Engineering Module (without heavy dependencies)
For demonstration and testing purposes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List
import re


class TargetEncoder:
    """Handles target variable encoding and rare class merging"""
    
    def __init__(self, rare_threshold: int = 5):
        self.rare_threshold = rare_threshold
        self.label_encoder = LabelEncoder()
        self.rare_classes = set()
    
    def create_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combined Category:Misconception target column"""
        df = df.copy()
        df["Category:Misconception"] = df.apply(
            lambda row: f"{row['Category']}:{row['Misconception']}" 
            if pd.notna(row["Misconception"]) 
            else f"{row['Category']}:NA",
            axis=1
        )
        return df
    
    def merge_rare_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge rare classes into 'Rare' category"""
        df = df.copy()
        counts = df["Category:Misconception"].value_counts()
        self.rare_classes = set(counts[counts <= self.rare_threshold].index)
        
        df.loc[df["Category:Misconception"].isin(self.rare_classes), 
               "Category:Misconception"] = "Rare"
        return df
    
    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Encode target variable"""
        df = df.copy()
        df["target"] = self.label_encoder.fit_transform(df["Category:Misconception"])
        return df, df["target"].values


class TextFeatureExtractor:
    """Extracts handcrafted features from text data"""
    
    @staticmethod
    def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive text-based features"""
        df = df.copy()
        
        # Basic length features
        df["exp_len"] = df["StudentExplanation"].astype(str).apply(len)
        df["exp_words"] = df["StudentExplanation"].astype(str).apply(lambda x: len(x.split()))
        df["q_len"] = df["QuestionText"].astype(str).apply(len)
        df["mc_len"] = df["MC_Answer"].astype(str).apply(len)
        
        # Mathematical symbols
        df["has_fraction"] = df["StudentExplanation"].str.contains("/", regex=False).astype(int)
        df["has_equals"] = df["StudentExplanation"].str.contains("=", regex=False).astype(int)
        df["has_percent"] = df["StudentExplanation"].str.contains("%", regex=False).astype(int)
        df["has_number"] = df["StudentExplanation"].str.contains(r"\d", regex=True).astype(int)
        
        # Mathematical operations
        df["has_plus"] = df["StudentExplanation"].str.contains(r"\+", regex=True).astype(int)
        df["has_minus"] = df["StudentExplanation"].str.contains(r"-", regex=True).astype(int)
        df["has_multiply"] = df["StudentExplanation"].str.contains(r"\*|×", regex=True).astype(int)
        df["has_divide"] = df["StudentExplanation"].str.contains(r"÷|/", regex=True).astype(int)
        
        # Text quality indicators
        df["exp_q_ratio"] = df["exp_len"] / (df["q_len"] + 1)
        df["is_exp_empty"] = (df["StudentExplanation"].astype(str).str.strip() == "").astype(int)
        
        # Count specific mathematical terms
        df["decimal_count"] = df["StudentExplanation"].str.count(r"\d+\.\d+")
        df["parentheses_count"] = df["StudentExplanation"].str.count(r"\(|\)")
        
        # Sentiment/confidence indicators
        df["has_question_mark"] = df["StudentExplanation"].str.contains(r"\?", regex=True).astype(int)
        df["has_exclamation"] = df["StudentExplanation"].str.contains(r"!", regex=True).astype(int)
        df["uncertainty_words"] = df["StudentExplanation"].str.count(
            r"\b(maybe|perhaps|might|could|unsure|think|guess)\b", flags=re.IGNORECASE
        )
        
        return df
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all handcrafted feature names"""
        return [
            "exp_len", "exp_words", "q_len", "mc_len",
            "has_fraction", "has_equals", "has_percent", "has_number",
            "has_plus", "has_minus", "has_multiply", "has_divide",
            "exp_q_ratio", "is_exp_empty", "decimal_count", "parentheses_count",
            "has_question_mark", "has_exclamation", "uncertainty_words"
        ]