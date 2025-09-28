import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb

# -------------------- Config --------------------
DATA_DIR = "/kaggle/input/map-charting-student-math-misunderstandings"
BGE_PATH = "/kaggle/input/bge-large-en/pytorch/default/1/bge-large-en"  # 按需修改
N_SPLITS = 5
RANDOM_STATE = 42

# Embedding / DR sizes
SVD_DIM = 128
PCA_DIM = 64   # 与 SVD 拼接，增加表征多样性

# XGBoost base params (可在此微调或用 optuna)
XGB_PARAMS = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=None,  # later set
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=1.5,
    min_child_weight=2,
    tree_method="gpu_hist",      # 如无 GPU，请改为 "hist"
    predictor="gpu_predictor",
    random_state=RANDOM_STATE
)

# -------------------- Load --------------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# -------------------- Target processing --------------------
train["Category:Misconception"] = train.apply(
    lambda r: f"{r['Category']}:{r['Misconception']}" if pd.notna(r["Misconception"]) else f"{r['Category']}:NA",
    axis=1
)

# 合并稀有类（阈值可调）
RARE_THRESH = 5
vc = train["Category:Misconception"].value_counts()
rare = vc[vc <= RARE_THRESH].index
train.loc[train["Category:Misconception"].isin(rare), "Category:Misconception"] = "Rare"

le = LabelEncoder()
train["target"] = le.fit_transform(train["Category:Misconception"])
y = train["target"].values
NUM_CLASSES = len(le.classes_)
XGB_PARAMS["num_class"] = NUM_CLASSES

# 类别权重 -> 样本权重
classes = np.unique(y)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
sample_weights = class_weights[y]

# -------------------- Embeddings --------------------
print("🔄 Loading embedding model and encoding texts ...")
embed_model = SentenceTransformer(BGE_PATH)

def concat_texts(df):
    return (df["QuestionText"].astype(str) + " " +
            df["MC_Answer"].astype(str) + " " +
            df["StudentExplanation"].astype(str)).tolist()

train_texts = concat_texts(train)
test_texts  = concat_texts(test)

X_train_embed = embed_model.encode(train_texts, show_progress_bar=True, batch_size=32)
X_test_embed  = embed_model.encode(test_texts, show_progress_bar=True, batch_size=32)

# 单独编码 Q/A/E 用于相似度/内积特征
q_train = embed_model.encode(train["QuestionText"].astype(str).tolist(), batch_size=32)
a_train = embed_model.encode(train["MC_Answer"].astype(str).tolist(), batch_size=32)
e_train = embed_model.encode(train["StudentExplanation"].astype(str).tolist(), batch_size=32)

q_test = embed_model.encode(test["QuestionText"].astype(str).tolist(), batch_size=32)
a_test = embed_model.encode(test["MC_Answer"].astype(str).tolist(), batch_size=32)
e_test = embed_model.encode(test["StudentExplanation"].astype(str).tolist(), batch_size=32)

# -------------------- Advanced Handcrafted Features --------------------
def jaccard(a, b):
    a_set, b_set = set(str(a).split()), set(str(b).split())
    if len(a_set) == 0 or len(b_set) == 0:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)

math_power_pattern = re.compile(r"\^|\*\*|\bsq\b|\bsquared\b", flags=re.I)
root_pattern = re.compile(r"sqrt|√", flags=re.I)
pi_pattern = re.compile(r"\b(pi|π)\b", flags=re.I)

def add_extra_features(df):
    df = df.copy()
    # length features
    df["exp_len"] = df["StudentExplanation"].astype(str).apply(len)
    df["exp_words"] = df["StudentExplanation"].astype(str).apply(lambda x: len(x.split()))
    df["q_len"] = df["QuestionText"].astype(str).apply(len)
    df["mc_len"] = df["MC_Answer"].astype(str).apply(len)
    df["exp_q_ratio"] = df["exp_len"] / (df["q_len"] + 1)
    df["is_exp_empty"] = (df["StudentExplanation"].astype(str).str.strip() == "").astype(int)

    # punctuation / symbol features
    df["has_fraction"] = df["StudentExplanation"].astype(str).str.contains("/", regex=False).astype(int)
    df["has_equals"] = df["StudentExplanation"].astype(str).str.contains("=", regex=False).astype(int)
    df["has_percent"] = df["StudentExplanation"].astype(str).str.contains("%", regex=False).astype(int)
    df["has_number"] = df["StudentExplanation"].astype(str).str.contains(r"\d", regex=True).astype(int)
    df["symbol_count"] = df["StudentExplanation"].apply(lambda x: sum(c in "+-*/=%^√" for c in str(x)))
    df["has_power"] = df["StudentExplanation"].astype(str).str.contains(math_power_pattern).astype(int)
    df["has_root"] = df["StudentExplanation"].astype(str).str.contains(root_pattern).astype(int)
    df["has_pi"] = df["StudentExplanation"].astype(str).str.contains(pi_pattern).astype(int)

    # lexical diversity
    df["word_diversity"] = df["StudentExplanation"].apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1))
    df["avg_word_len"] = df["StudentExplanation"].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
    df["stopword_ratio"] = df["StudentExplanation"].apply(lambda x: sum(w in ENGLISH_STOP_WORDS for w in str(x).lower().split()) / (len(str(x).split()) + 1))

    # case / uppercase ratio
    df["uppercase_ratio"] = df["StudentExplanation"].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))

    # repeated words ratio
    def repeat_ratio(s):
        toks = [w for w in str(s).lower().split() if w]
        if not toks:
            return 0.0
        return 1 - len(set(toks)) / len(toks)
    df["repeat_word_ratio"] = df["StudentExplanation"].apply(repeat_ratio)

    # overlap features
    df["overlap_q"] = df.apply(lambda r: jaccard(r["QuestionText"], r["StudentExplanation"]), axis=1)
    df["overlap_mc"] = df.apply(lambda r: jaccard(r["MC_Answer"], r["StudentExplanation"]), axis=1)
    df["qa_overlap"] = df.apply(lambda r: jaccard(r["QuestionText"], r["MC_Answer"]), axis=1)

    df["exp_q_diff"] = (df["exp_len"] - df["q_len"]).abs()
    df["exp_mc_diff"] = (df["exp_len"] - df["mc_len"]).abs()

    return df

train = add_extra_features(train)
test  = add_extra_features(test)

# -------------------- Embedding similarity features --------------------
def compute_pair_sim(q_embed, a_embed, e_embed):
    qe = [cosine_similarity([q],[e])[0,0] for q,e in zip(q_embed,e_embed)]
    ae = [cosine_similarity([a],[e])[0,0] for a,e in zip(a_embed,e_embed)]
    qa = [cosine_similarity([q],[a])[0,0] for q,a in zip(q_embed,a_embed)]
    # also inner product (dot)
    qdote = [float(np.dot(q, e)) for q,e in zip(q_embed,e_embed)]
    adote = [float(np.dot(a, e)) for a,e in zip(a_embed,e_embed)]
    return np.array(qe), np.array(ae), np.array(qa), np.array(qdote), np.array(adote)

train_qe, train_ae, train_qa, train_qdot_e, train_adot_e = compute_pair_sim(q_train, a_train, e_train)
test_qe, test_ae, test_qa, test_qdot_e, test_adot_e = compute_pair_sim(q_test, a_test, e_test)

train["qe_sim"] = train_qe
train["ae_sim"] = train_ae
train["qa_sim"] = train_qa
train["qdot_e"] = train_qdot_e
train["adot_e"] = train_adot_e

test["qe_sim"] = test_qe
test["ae_sim"] = test_ae
test["qa_sim"] = test_qa
test["qdot_e"] = test_qdot_e
test["adot_e"] = test_adot_e

train["sim_diff"] = train["qe_sim"] - train["ae_sim"]
test["sim_diff"]  = test["qe_sim"] - test["ae_sim"]
train["qae_balance"] = (train["qe_sim"] + train["ae_sim"]) / 2 - train["qa_sim"]
test["qae_balance"]  = (test["qe_sim"] + test["ae_sim"]) / 2 - test["qa_sim"]

# -------------------- TF-IDF based similarity features --------------------
print("🔎 Computing TF-IDF similarities ...")
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
# fit on combined text to avoid unseen during test
all_texts = (train["QuestionText"].astype(str).tolist() +
             train["MC_Answer"].astype(str).tolist() +
             train["StudentExplanation"].astype(str).tolist() +
             test["QuestionText"].astype(str).tolist() +
             test["MC_Answer"].astype(str).tolist() +
             test["StudentExplanation"].astype(str).tolist())

tfidf.fit(all_texts)

def tfidf_sim_series(df):
    q_mat = tfidf.transform(df["QuestionText"].astype(str).tolist())
    a_mat = tfidf.transform(df["MC_Answer"].astype(str).tolist())
    e_mat = tfidf.transform(df["StudentExplanation"].astype(str).tolist())

    # cosine sim for each row
    sims_qe = np.array([cosine_similarity(q_mat[i], e_mat[i])[0,0] for i in range(q_mat.shape[0])])
    sims_ae = np.array([cosine_similarity(a_mat[i], e_mat[i])[0,0] for i in range(a_mat.shape[0])])
    sims_qa = np.array([cosine_similarity(q_mat[i], a_mat[i])[0,0] for i in range(q_mat.shape[0])])
    return sims_qe, sims_ae, sims_qa

train_tfidf_qe, train_tfidf_ae, train_tfidf_qa = tfidf_sim_series(train)
test_tfidf_qe, test_tfidf_ae, test_tfidf_qa = tfidf_sim_series(test)

train["tfidf_qe"] = train_tfidf_qe
train["tfidf_ae"] = train_tfidf_ae
train["tfidf_qa"] = train_tfidf_qa
test["tfidf_qe"] = test_tfidf_qe
test["tfidf_ae"] = test_tfidf_ae
test["tfidf_qa"] = test_tfidf_qa

# -------------------- Collect feature list --------------------
extra_feats = [
    "exp_len","exp_words","has_fraction","has_equals","has_percent","has_number",
    "q_len","mc_len","exp_q_ratio","is_exp_empty","symbol_count","has_power","has_root","has_pi",
    "word_diversity","avg_word_len","stopword_ratio","uppercase_ratio","repeat_word_ratio",
    "overlap_q","overlap_mc","qa_overlap","exp_q_diff","exp_mc_diff",
    "qe_sim","ae_sim","qa_sim","qdot_e","adot_e","sim_diff","qae_balance",
    "tfidf_qe","tfidf_ae","tfidf_qa"
]

# ensure no missing values
for c in extra_feats:
    if c not in train.columns:
        train[c] = 0
    if c not in test.columns:
        test[c] = 0

# Standardize extra features
scaler = StandardScaler()
train_extra_scaled = scaler.fit_transform(train[extra_feats].fillna(0).values)
test_extra_scaled  = scaler.transform(test[extra_feats].fillna(0).values)

# -------------------- Dimensionality reduction: SVD + PCA --------------------
print("⚡ Reducing embedding dimensions (SVD + PCA) ...")
svd = TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_STATE)
X_train_svd = svd.fit_transform(X_train_embed)
X_test_svd  = svd.transform(X_test_embed)

pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_embed)
X_test_pca  = pca.transform(X_test_embed)

# 拼接 embedding 降维 + extra features
X_train_full = np.hstack([X_train_svd, X_train_pca, train_extra_scaled])
X_test_full  = np.hstack([X_test_svd,  X_test_pca,  test_extra_scaled])

print("X_train_full shape:", X_train_full.shape)
print("X_test_full shape:", X_test_full.shape)

# -------------------- CV training with stacking --------------------
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((X_train_full.shape[0], NUM_CLASSES))
test_preds = np.zeros((X_test_full.shape[0], NUM_CLASSES))

# For stacking: collect base model OOF predictions (for meta model)
oof_meta = np.zeros((X_train_full.shape[0], NUM_CLASSES))
test_meta = np.zeros((X_test_full.shape[0], NUM_CLASSES))

val_loglosses = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full, y), 1):
    print(f"\n--- Fold {fold} ---")
    X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    w_tr = sample_weights[tr_idx]

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )

    # predict
    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test_full)

    oof_preds[val_idx] = val_proba
    test_preds += test_proba / N_SPLITS

    # for meta
    oof_meta[val_idx] = val_proba
    test_meta += test_proba / N_SPLITS

    # compute fold logloss (use best iteration from evals_result)
    try:
        res = model.evals_result()
        best_it = model.best_iteration
        fold_logloss = res["validation_0"]["mlogloss"][best_it]
    except Exception:
        # fallback compute logloss directly
        fold_logloss = log_loss(y_val, val_proba)
    print(f"Fold {fold} logloss: {fold_logloss:.6f}")
    val_loglosses.append(fold_logloss)

print("\n📊 CV mean logloss:", np.mean(val_loglosses))

# -------------------- Stacking: train simple meta model on OOF preds --------------------
print("🔁 Training meta model (LogisticRegression) on OOF predictions ...")
meta = LogisticRegression(max_iter=2000, multi_class="multinomial")
meta.fit(oof_meta, y)

# meta predictions on test
meta_test_proba = meta.predict_proba(test_meta)

# choose final: average base ensemble + meta (simple blend)
FINAL_PROBA = (test_preds * 0.6) + (meta_test_proba * 0.4)

# -------------------- Prepare submission --------------------
# 确认 sample_submission.csv 的格式
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print("📑 Sample submission columns:", sample_sub.columns.tolist())

# 转换预测为标签
test_preds_labels = np.argmax(FINAL_PROBA, axis=1)
test_labels = le.inverse_transform(test_preds_labels)

# 构造提交文件
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": test_labels
})

# 确保列顺序一致
submission = submission[sample_sub.columns]

# 保存提交文件
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved: submission.csv")

# 打印前几行检查
print(submission.head())