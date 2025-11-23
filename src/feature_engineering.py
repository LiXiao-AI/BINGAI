"""
Feature Engineering Module for Math Misconception Classification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
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


class EmbeddingProcessor:
    """Handles text embedding generation and processing"""
    
    def __init__(self, model_path: str, batch_size: int = 64):
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = None
        self.svd = None
    
    def load_model(self):
        """Load the sentence transformer model"""
        print(f"🔄 Loading embedding model from {self.model_path}")
        self.model = SentenceTransformer(self.model_path)
    
    def create_combined_text(self, df: pd.DataFrame) -> List[str]:
        """Combine text columns for embedding"""
        return (df["QuestionText"].astype(str) + " " +
                df["MC_Answer"].astype(str) + " " +
                df["StudentExplanation"].astype(str)).tolist()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text data"""
        if self.model is None:
            self.load_model()
        
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=self.batch_size
        )
        return embeddings
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 128) -> np.ndarray:
        """Reduce embedding dimensions using TruncatedSVD"""
        print(f"⚡ Reducing embedding dimensions to {n_components}...")
        if self.svd is None:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_embeddings = self.svd.fit_transform(embeddings)
        else:
            reduced_embeddings = self.svd.transform(embeddings)
        
        print(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
        return reduced_embeddings


class FeaturePipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.target_encoder = TargetEncoder(config.RARE_CLASS_THRESHOLD)
        self.text_extractor = TextFeatureExtractor()
        self.embedding_processor = EmbeddingProcessor(
            config.MODEL_PATH, 
            config.EMBEDDING_BATCH_SIZE
        )
        self.scaler = StandardScaler()
        
    def process_training_data(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
        """Process training data through complete pipeline"""
        print("🚀 Processing training data...")
        
        # Target encoding
        train_df = self.target_encoder.create_target_column(train_df)
        train_df = self.target_encoder.merge_rare_classes(train_df)
        train_df, y = self.target_encoder.encode_target(train_df)
        
        # Text features
        train_df = self.text_extractor.add_text_features(train_df)
        feature_names = self.text_extractor.get_feature_names()
        
        # Scale handcrafted features
        train_extra_scaled = self.scaler.fit_transform(train_df[feature_names].values)
        
        # Generate and reduce embeddings
        train_texts = self.embedding_processor.create_combined_text(train_df)
        train_embeddings = self.embedding_processor.generate_embeddings(train_texts)
        train_embeddings_reduced = self.embedding_processor.reduce_dimensions(
            train_embeddings, self.config.SVD_COMPONENTS
        )
        
        # Combine features
        X_train = np.hstack([train_embeddings_reduced, train_extra_scaled])
        
        print(f"✅ Training data processed: {X_train.shape}")
        return X_train, y, self.target_encoder.label_encoder
    
    def process_test_data(self, test_df: pd.DataFrame) -> np.ndarray:
        """Process test data through pipeline"""
        print("🚀 Processing test data...")
        
        # Text features
        test_df = self.text_extractor.add_text_features(test_df)
        feature_names = self.text_extractor.get_feature_names()
        
        # Scale handcrafted features
        test_extra_scaled = self.scaler.transform(test_df[feature_names].values)
        
        # Generate and reduce embeddings
        test_texts = self.embedding_processor.create_combined_text(test_df)
        test_embeddings = self.embedding_processor.generate_embeddings(test_texts)
        test_embeddings_reduced = self.embedding_processor.reduce_dimensions(
            test_embeddings, self.config.SVD_COMPONENTS
        )
        
        # Combine features
        X_test = np.hstack([test_embeddings_reduced, test_extra_scaled])
        
        print(f"✅ Test data processed: {X_test.shape}")
        return X_test