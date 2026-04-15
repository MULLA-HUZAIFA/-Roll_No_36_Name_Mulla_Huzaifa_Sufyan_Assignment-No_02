# Sentiment Analysis on Oppenheimer Movie Tweets

## Assignment 2 – Text Analytics | Roll No 36 | Mulla Huzaifa Sufyan

---

## 📌 Overview

This project performs **sentiment analysis** on 100 manually labelled tweets about the movie **Oppenheimer (2023)**. Three machine learning classifiers are applied and compared:

- **Naïve Bayes**
- **Support Vector Machine (SVM)**
- **Logistic Regression**

---

## 🗂️ Repository Structure

```
RollNo_36_Name_ Mulla_Huzaifa_Sufyan_Assignment2/
│
├── data/
│   ├── tweets_dataset.csv      ← Full 100 labelled tweets
│   ├── train_data.csv          ← 80 training tweets
│   └── test_data.csv           ← 20 test tweets
│
├── notebooks/
│   ├── Sentiment_Analysis_Assignment2.ipynb  ← Main Jupyter Notebook
│   └── sentiment_analysis.py                 ← Standalone Python script
│
├── results/
│   └── model_results.csv       ← Precision, Recall, F1 summary
│
├── visualizations/
│   ├── label_distribution.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── per_class_heatmap.png
│   └── tweet_length_dist.png
│
├── reports/
│   └── Assignment2_Report.pdf  ← Final report
│
└── README.md
```

---

## 🎯 Topic

**Movie: Oppenheimer (2023)**  
Tweets were manually collected and labelled as:

- **Positive** – praise, excitement, 5-star reactions
- **Neutral** – balanced, neither positive nor negative
- **Negative** – criticism, disappointment, dislike

---

## 📊 Dataset

| Split     | Size    | Positive | Neutral | Negative |
| --------- | ------- | -------- | ------- | -------- |
| Train     | 80      | 32       | 24      | 24       |
| Test      | 20      | 8        | 6       | 6        |
| **Total** | **100** | **40**   | **30**  | **30**   |

Labels were **manually assigned** based on the sentiment expressed in each tweet.

---

## ⚙️ Methodology

1. **Text Preprocessing** – lowercase, remove URLs, punctuation, special chars
2. **Vectorisation** – TF-IDF with unigrams & bigrams (`ngram_range=(1,2)`)
3. **Train/Test Split** – 80% train, 20% test (stratified)
4. **Classification** – Naïve Bayes, SVM (LinearSVC), Logistic Regression
5. **Evaluation** – Precision, Recall, F1-Score, Accuracy + Confusion Matrices

---

## 📈 Results

| Model               | Accuracy | Precision  | Recall   | F1-Score   |
| ------------------- | -------- | ---------- | -------- | ---------- |
| Naïve Bayes         | 0.45     | 0.4367     | 0.45     | 0.3851     |
| **SVM**             | **0.45** | **0.4596** | **0.45** | **0.4219** |
| Logistic Regression | 0.45     | 0.4778     | 0.45     | 0.3319     |

🏆 **Best Model:** SVM (highest F1-Score)

> _Note: Accuracy is limited by the small 20-sample test set. Results are expected to improve significantly with more data._

---

## 🖼️ Visualizations

| Plot                     | Description                          |
| ------------------------ | ------------------------------------ |
| `label_distribution.png` | Pie chart of sentiment labels        |
| `confusion_matrices.png` | Confusion matrix for all 3 models    |
| `model_comparison.png`   | Side-by-side bar chart comparison    |
| `per_class_heatmap.png`  | Per-class precision & recall heatmap |
| `tweet_length_dist.png`  | Word count distribution by sentiment |

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/shoaibjamadar-oss/Roll_No_36_Name_Mulla_Huzaifa_Sufyan_Assigment-No_02.git
cd Roll_No_22_Name_ Mulla_Huzaifa_Sufyan_Assigment-No_02

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# Run the script
python notebooks/sentiment_analysis.py

# Or open the notebook
jupyter notebook notebooks/Sentiment_Analysis_Assignment2.ipynb
```

---

## 📦 Requirements

```
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
numpy>=1.21.0
```

---

## 📋 Key Findings

- **SVM** achieved the best F1-Score (0.4219) — ideal for high-dimensional TF-IDF features
- **Logistic Regression** achieved highest precision but lower F1 due to class imbalance on neutral
- **Naïve Bayes** was the simplest and fastest with competitive performance
- The **neutral class** is consistently the most difficult to classify
- Bigram features helped capture phrases like "fell asleep" and "absolute masterpiece"

---

## 👤 Author

| Field      | Details                       |
| ---------- | ----------------------------- |
| Name       | Mulla Huzaifa Sufyan          |
| Roll No    | 36                            |
| Assignment | Assignment 2 – Text Analytics |
| Topic      | Oppenheimer Movie Tweets      |
