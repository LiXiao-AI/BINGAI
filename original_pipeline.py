"""
Original Pipeline Script - Direct translation of the provided code
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# -------------------- Step 1: Load --------------------
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df  = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# -------------------- Step 2: Target (merge rare classes) --------------------
train_df["Category:Misconception"] = train_df.apply(
    lambda row: f"{row['Category']}:{row['Misconception']}" if pd.notna(row["Misconception"]) else f"{row['Category']}:NA",
    axis=1
)

# 合并极少类别
counts = train_df["Category:Misconception"].value_counts()
rare_classes = counts[counts <= 5].index
train_df.loc[train_df["Category:Misconception"].isin(rare_classes), "Category:Misconception"] = "Rare"

label_encoder = LabelEncoder()
train_df["target"] = label_encoder.fit_transform(train_df["Category:Misconception"])
y = train_df["target"].values

# 计算类别权重（缓解类别不平衡）
classes = np.unique(y)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
sample_weights = class_weights[y]   # 每个样本对应的权重

# -------------------- Step 3: Text Embedding --------------------
print("🔄 Encoding texts with StudentExplanation included ...")
MODEL_PATH = "/kaggle/input/bge_embeddings/pytorch/default/1/finetuned_bge_embeddings_v5_small_v1.5"
embed_model = SentenceTransformer(MODEL_PATH)

train_texts = (train_df["QuestionText"].astype(str) + " " +
               train_df["MC_Answer"].astype(str) + " " +
               train_df["StudentExplanation"].astype(str)).tolist()
test_texts  = (test_df["QuestionText"].astype(str) + " " +
               test_df["MC_Answer"].astype(str) + " " +
               test_df["StudentExplanation"].astype(str)).tolist()

X_train_embed = embed_model.encode(train_texts, show_progress_bar=True, batch_size=64)
X_test_embed  = embed_model.encode(test_texts, show_progress_bar=True, batch_size=64)

# -------------------- Step 3.5: Add handcrafted features --------------------
def add_features(df):
    # 原有
    df["exp_len"] = df["StudentExplanation"].astype(str).apply(len)
    df["exp_words"] = df["StudentExplanation"].astype(str).apply(lambda x: len(x.split()))
    df["has_fraction"] = df["StudentExplanation"].str.contains("/", regex=False).astype(int)

    # 数学符号相关
    df["has_equals"] = df["StudentExplanation"].str.contains("=", regex=False).astype(int)
    df["has_percent"] = df["StudentExplanation"].str.contains("%", regex=False).astype(int)
    df["has_number"] = df["StudentExplanation"].str.contains(r"\d", regex=True).astype(int)

    # 文本质量
    df["q_len"] = df["QuestionText"].astype(str).apply(len)
    df["mc_len"] = df["MC_Answer"].astype(str).apply(len)
    df["exp_q_ratio"] = df["exp_len"] / (df["q_len"] + 1)

    # 缺失值
    df["is_exp_empty"] = (df["StudentExplanation"].astype(str).str.strip() == "").astype(int)

    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

extra_feats = [
    "exp_len", "exp_words", "has_fraction",
    "has_equals", "has_percent", "has_number",
    "q_len", "mc_len", "exp_q_ratio", "is_exp_empty"
]

# 标准化手工特征（防止尺度差异）
scaler = StandardScaler()
train_extra_scaled = scaler.fit_transform(train_df[extra_feats].values)
test_extra_scaled = scaler.transform(test_df[extra_feats].values)

# -------------------- Step 3.6: Dimensionality Reduction --------------------
print("⚡ Reducing embedding dimensions to 128 ...")
svd = TruncatedSVD(n_components=128, random_state=42)
X_train_svd = svd.fit_transform(X_train_embed)
X_test_svd  = svd.transform(X_test_embed)

# 拼接 embedding + 标准化特征
X_train_full = np.hstack([X_train_svd, train_extra_scaled])
X_test_full  = np.hstack([X_test_svd,  test_extra_scaled])

# -------------------- Step 4: StratifiedKFold Training --------------------
print("🚀 Start StratifiedKFold training and averaging probabilities ...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_pred_proba = np.zeros((X_test_full.shape[0], len(label_encoder.classes_)))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y), 1):
    print(f"\n--- Fold {fold} ---")
    X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    w_tr = sample_weights[train_idx]  # 对应的样本权重

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        n_estimators=3000,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=1.5,
        min_child_weight=2,
        tree_method="gpu_hist",    # 🚀 GPU 加速
        predictor="gpu_predictor", # 🚀 GPU 预测
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )

    best_iter = model.best_iteration
    val_logloss = model.evals_result()["validation_0"]["mlogloss"][best_iter]
    val_scores.append(val_logloss)
    print(f"Fold {fold} best logloss: {val_logloss:.4f}")

    test_pred_proba += model.predict_proba(X_test_full) / kf.n_splits

print("\n📊 CV logloss:", np.mean(val_scores))

# -------------------- Step 5: Predict --------------------
test_preds = np.argmax(test_pred_proba, axis=1)
test_labels = label_encoder.inverse_transform(test_preds)

submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "Category:Misconception": test_labels
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved: submission.csv")