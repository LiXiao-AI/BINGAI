# Math Misconception Classification Pipeline

A comprehensive machine learning pipeline for classifying student math misconceptions using text embeddings and XGBoost. This project is designed for the Kaggle competition "Map: Charting Student Math Misunderstandings".

## 🚀 Features

- **Advanced Text Processing**: Combines question text, multiple choice answers, and student explanations
- **Sentence Embeddings**: Uses fine-tuned BGE embeddings for semantic understanding
- **Feature Engineering**: Comprehensive handcrafted features for mathematical content analysis
- **Ensemble Learning**: 5-fold cross-validation with XGBoost for robust predictions
- **Class Balancing**: Handles imbalanced classes with computed sample weights
- **Modular Design**: Clean, maintainable code structure with separate modules

## 📁 Project Structure

```
BINGAI/
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── feature_engineering.py # Feature extraction and processing
│   └── model_training.py      # Model training and prediction
├── data/                      # Data directory (for local development)
├── models/                    # Saved models directory
├── outputs/                   # Output files (submissions, analysis)
├── main.py                    # Main pipeline script
├── original_pipeline.py       # Original monolithic script
├── test_simple.py            # Basic component tests
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BINGAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### For Kaggle Competition

The pipeline is designed to work directly in Kaggle environments:

```python
python main.py
```

### For Local Development

1. Update data paths in `src/config.py` or the pipeline will automatically detect local environment
2. Place your data files in the `data/` directory:
   - `train.csv`
   - `test.csv`
3. Download the BGE embedding model to `models/sentence_transformer_model/`

### Running the Original Script

For comparison with the original implementation:

```python
python original_pipeline.py
```

## 🔧 Configuration

Key parameters can be adjusted in `src/config.py`:

- **Model Parameters**: XGBoost hyperparameters, embedding dimensions
- **Cross-Validation**: Number of folds, random seeds
- **Feature Engineering**: Rare class threshold, text processing options
- **Paths**: Data and model file locations

## 🧪 Testing

Run basic component tests:

```bash
python test_simple.py
```

This verifies that core pipeline components work correctly without requiring heavy dependencies.

## 📈 Pipeline Overview

### 1. Data Loading & Target Engineering
- Combines Category and Misconception into unified target
- Merges rare classes (≤5 samples) into "Rare" category
- Applies label encoding for multi-class classification

### 2. Feature Engineering

#### Text Embeddings
- Concatenates QuestionText + MC_Answer + StudentExplanation
- Generates embeddings using fine-tuned BGE model
- Reduces dimensions to 128 using TruncatedSVD

#### Handcrafted Features
- **Length Features**: Text length, word count ratios
- **Mathematical Symbols**: Fractions, equals, percentages, numbers
- **Mathematical Operations**: Plus, minus, multiply, divide
- **Quality Indicators**: Empty explanations, question marks
- **Confidence Markers**: Uncertainty words, exclamations

### 3. Model Training
- **Algorithm**: XGBoost with GPU acceleration
- **Cross-Validation**: 5-fold stratified splits
- **Class Balancing**: Computed sample weights
- **Early Stopping**: Prevents overfitting
- **Ensemble**: Averages predictions across folds

### 4. Prediction & Output
- Generates probability distributions
- Creates submission file with predicted categories
- Provides prediction analysis and distribution statistics

## 🎯 Key Improvements Over Original

1. **Modular Architecture**: Separated concerns into logical modules
2. **Enhanced Features**: Added more mathematical and linguistic features
3. **Better Configuration**: Centralized parameter management
4. **Error Handling**: Robust path detection and validation
5. **Testing**: Component verification and validation
6. **Documentation**: Comprehensive code documentation
7. **Extensibility**: Easy to add new features or models

## 📋 Feature List

### Text-Based Features
- `exp_len`, `exp_words`: Explanation length metrics
- `q_len`, `mc_len`: Question and answer length
- `exp_q_ratio`: Explanation to question length ratio

### Mathematical Content
- `has_fraction`, `has_equals`, `has_percent`: Symbol presence
- `has_number`: Numeric content detection
- `has_plus`, `has_minus`, `has_multiply`, `has_divide`: Operations
- `decimal_count`, `parentheses_count`: Mathematical notation

### Quality & Confidence
- `is_exp_empty`: Empty explanation detection
- `has_question_mark`, `has_exclamation`: Punctuation analysis
- `uncertainty_words`: Confidence level indicators

## 🚀 Performance

The pipeline achieves competitive performance through:
- **Semantic Understanding**: BGE embeddings capture mathematical concepts
- **Feature Diversity**: Combines embeddings with domain-specific features
- **Ensemble Robustness**: Cross-validation reduces overfitting
- **Class Balance**: Weighted training handles imbalanced data

## 🔮 Future Enhancements

- **Advanced Embeddings**: Experiment with mathematical domain-specific models
- **Feature Selection**: Automated feature importance analysis
- **Model Ensemble**: Combine multiple algorithms (LightGBM, CatBoost)
- **Hyperparameter Tuning**: Automated optimization with Optuna
- **Error Analysis**: Detailed misconception pattern analysis

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
