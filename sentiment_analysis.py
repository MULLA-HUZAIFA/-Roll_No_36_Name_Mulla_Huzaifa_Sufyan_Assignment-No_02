"""
=====================================================
Assignment 2: Text Analytics – Sentiment Analysis
Topic: Oppenheimer (Movie)
Roll No: 36 | Name: Mulla Huzaifa Sufyan
=====================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "tweets_dataset.csv")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
VIZ_DIR      = os.path.join(BASE_DIR, "visualizations")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# ─── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {"positive": "#4CAF50", "neutral": "#FF9800", "negative": "#F44336"}
MODEL_COLOURS = ["#3B82F6", "#8B5CF6", "#EC4899"]
FIG_BG  = "#0F172A"
TEXT_COL = "#E2E8F0"

def apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  FIG_BG,
        'axes.facecolor':    '#1E293B',
        'axes.edgecolor':    '#334155',
        'axes.labelcolor':   TEXT_COL,
        'xtick.color':       TEXT_COL,
        'ytick.color':       TEXT_COL,
        'text.color':        TEXT_COL,
        'grid.color':        '#334155',
        'grid.alpha':        0.5,
        'font.family':       'DejaVu Sans',
    })

apply_dark_style()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SENTIMENT ANALYSIS – OPPENHEIMER TWEETS")
print("  Roll No 22 | Jamadar Shoaib Anwar")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"\n✔  Loaded {len(df)} tweets")
print(df['label'].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'@\w+', '', text)                     # remove mentions
    text = re.sub(r'#\w+', '', text)                     # remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)                 # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()             # normalise spaces
    return text

df['clean_tweet'] = df['tweet'].apply(clean_text)

# Encode labels
label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
df['label_enc'] = df['label'].map(label_map)

print(f"\n✔  Text cleaned. Sample:")
for _, row in df.sample(3, random_state=42).iterrows():
    print(f"   [{row['label']}]  {row['clean_tweet'][:70]}…")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT  (80 / 20)
# ═══════════════════════════════════════════════════════════════════════════════
X = df['clean_tweet']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n✔  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ─── Save train/test splits ────────────────────────────────────────────────────
df_train = df.loc[X_train.index][['id','tweet','label']]
df_test  = df.loc[X_test.index][['id','tweet','label']]
df_train.to_csv(os.path.join(BASE_DIR, "data", "train_data.csv"), index=False)
df_test.to_csv(os.path.join(BASE_DIR,  "data", "test_data.csv"),  index=False)
print("✔  train_data.csv and test_data.csv saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. VECTORISATION (TF-IDF)
# ═══════════════════════════════════════════════════════════════════════════════
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"\n✔  TF-IDF vectorised  |  vocabulary size: {len(vectorizer.vocabulary_)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════════
models = {
    "Naïve Bayes":        MultinomialNB(alpha=0.5),
    "SVM":                LinearSVC(C=1.0, max_iter=2000, random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    prec  = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1    = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    acc   = accuracy_score(y_test, y_pred)
    cm    = confusion_matrix(y_test, y_pred, labels=['positive','neutral','negative'])
    report = classification_report(y_test, y_pred, zero_division=0)

    results[name] = {
        'precision': prec, 'recall': rec,
        'f1': f1, 'accuracy': acc,
        'confusion_matrix': cm,
        'report': report,
        'y_pred': y_pred,
    }
    print(f"\n── {name} ──")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 6a. Label distribution pie chart ──────────────────────────────────────────
counts = df['label'].value_counts()
fig, ax = plt.subplots(figsize=(7, 7), facecolor=FIG_BG)
wedge_props = dict(linewidth=2, edgecolor=FIG_BG)
colours = [PALETTE[l] for l in counts.index]
wedges, texts, autotexts = ax.pie(
    counts, labels=counts.index, autopct='%1.1f%%',
    colors=colours, wedgeprops=wedge_props,
    textprops={'color': TEXT_COL, 'fontsize': 13}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
ax.set_title('Tweet Label Distribution\n(Oppenheimer – 100 Tweets)',
             fontsize=15, fontweight='bold', color=TEXT_COL, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "label_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\n✔  label_distribution.png saved.")

# ── 6b. Confusion matrices ─────────────────────────────────────────────────────
classes = ['positive', 'neutral', 'negative']
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=FIG_BG)
fig.suptitle('Confusion Matrices', fontsize=17, fontweight='bold', color=TEXT_COL, y=1.02)

for ax, (name, res), col in zip(axes, results.items(), MODEL_COLOURS):
    cm_norm = res['confusion_matrix'].astype(float)
    cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=10, color=TEXT_COL)
    ax.set_yticklabels(classes, fontsize=10, color=TEXT_COL)
    ax.set_xlabel('Predicted', fontsize=11, color=TEXT_COL)
    ax.set_ylabel('Actual', fontsize=11, color=TEXT_COL)
    ax.set_title(name, fontsize=13, fontweight='bold', color=col, pad=10)
    for i in range(3):
        for j in range(3):
            val = res['confusion_matrix'][i, j]
            pct = cm_norm[i, j]
            colour = 'white' if pct > 0.5 else TEXT_COL
            ax.text(j, i, f"{val}\n({pct:.0%})", ha='center', va='center',
                    fontsize=11, color=colour, fontweight='bold')
    ax.set_facecolor('#1E293B')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✔  confusion_matrices.png saved.")

# ── 6c. Model comparison bar chart ────────────────────────────────────────────
metrics = ['precision', 'recall', 'f1', 'accuracy']
metric_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
model_names = list(results.keys())

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6), facecolor=FIG_BG)
ax.set_facecolor('#1E293B')
for i, (name, col) in enumerate(zip(model_names, MODEL_COLOURS)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=col, alpha=0.88, edgecolor=FIG_BG, linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.2f}', ha='center', va='bottom',
                fontsize=9, color=col, fontweight='bold')

ax.set_xlabel('Metric', fontsize=12, color=TEXT_COL, labelpad=10)
ax.set_ylabel('Score', fontsize=12, color=TEXT_COL, labelpad=10)
ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold',
             color=TEXT_COL, pad=15)
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(loc='upper right', facecolor='#1E293B', edgecolor='#334155',
          labelcolor=TEXT_COL, fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor('#334155')

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✔  model_comparison.png saved.")

# ── 6d. Per-class precision & recall heatmap ──────────────────────────────────
classes_list = ['positive', 'neutral', 'negative']
heatmap_data = {}
for name, res in results.items():
    row = {}
    y_pred = res['y_pred']
    p_per_class = precision_score(y_test, y_pred, labels=classes_list,
                                  average=None, zero_division=0)
    r_per_class = recall_score(y_test, y_pred, labels=classes_list,
                               average=None, zero_division=0)
    for cls, p, r in zip(classes_list, p_per_class, r_per_class):
        row[f'{cls}_P'] = round(p, 2)
        row[f'{cls}_R'] = round(r, 2)
    heatmap_data[name] = row

hdf = pd.DataFrame(heatmap_data).T

fig, ax = plt.subplots(figsize=(10, 4), facecolor=FIG_BG)
ax.set_facecolor('#1E293B')
sns.heatmap(hdf, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, linecolor='#0F172A',
            ax=ax, cbar_kws={'label': 'Score'})
ax.set_title('Per-Class Precision (P) & Recall (R) by Model',
             fontsize=13, fontweight='bold', color=TEXT_COL, pad=12)
ax.tick_params(colors=TEXT_COL)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "per_class_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✔  per_class_heatmap.png saved.")

# ── 6e. Tweet length distribution by sentiment ────────────────────────────────
df['tweet_length'] = df['tweet'].apply(lambda x: len(x.split()))
fig, ax = plt.subplots(figsize=(10, 5), facecolor=FIG_BG)
ax.set_facecolor('#1E293B')
for label, colour in PALETTE.items():
    subset = df[df['label'] == label]['tweet_length']
    ax.hist(subset, bins=15, alpha=0.65, label=label, color=colour, edgecolor=FIG_BG)
ax.set_xlabel('Tweet Word Count', fontsize=12, color=TEXT_COL)
ax.set_ylabel('Frequency', fontsize=12, color=TEXT_COL)
ax.set_title('Tweet Length Distribution by Sentiment', fontsize=14,
             fontweight='bold', color=TEXT_COL)
ax.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor=TEXT_COL)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor('#334155')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "tweet_length_dist.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✔  tweet_length_dist.png saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS CSV
# ═══════════════════════════════════════════════════════════════════════════════
summary_rows = []
for name, res in results.items():
    summary_rows.append({
        'Model': name,
        'Accuracy':  round(res['accuracy'],  4),
        'Precision': round(res['precision'], 4),
        'Recall':    round(res['recall'],    4),
        'F1-Score':  round(res['f1'],        4),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"), index=False)
print("\n✔  model_results.csv saved.")
print("\n" + summary_df.to_string(index=False))

# ─── Best model ───────────────────────────────────────────────────────────────
best = summary_df.loc[summary_df['F1-Score'].idxmax(), 'Model']
print(f"\n🏆  Best model by F1-Score: {best}")

print("\n" + "="*60)
print("  ALL DONE – visualisations & results written to disk.")
print("="*60 + "\n")
