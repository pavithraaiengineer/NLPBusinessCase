# NLPBusinessCase — Amazon Reviews Intelligence Platform

> **End-to-end NLP pipeline** that transforms raw Amazon Electronics reviews into actionable business intelligence using sentiment classification, unsupervised clustering, and abstractive summarisation.

---

##  Project Overview

This project applies state-of-the-art NLP techniques to the **McAuley-Lab Amazon Reviews 2023 (Electronics)** dataset.

| Business Question | NLP Solution |
|---|---|
| Is this review positive, neutral, or negative? | RoBERTa VADER fine-tuned 3-class classifier |
| What topics do customers talk about? | K-Means + TF-IDF / sentence embedding clustering |
| What's the gist of 1,000 reviews? | BART / T5 abstractive summarisation |
| Which products have mismatched ratings vs. sentiment? | VADER + star rating agreement score |

---

##  Repository Structure

```
NLPBusinessCase/
│__ Amazon_VADER_RoBERTa_Notebook.ipynb              # Phase 1 & 2: EDA + RoBERTa sentiment classifier
├── Data_preprocessing_classification_NLPLLM.ipynb   # Phase 1 & 2: EDA + RoBERTa sentiment classifier
├── Clustering_ProjectNLPLLM.ipynb                   # Phase 3: Unsupervised topic clustering
├── Summarization_Final.ipynb                        # Phase 4: Abstractive summarisation pipeline
├── ReviewIntel_App.html                             # Interactive front-end demo app
├── Reviews_classified_Roberta_VADER.csv             # Output: classified reviews with VADER scores
├── clustered_products.csv                           # Output: products with cluster labels
└── README.md
```

---

##  Notebooks

### `Data_preprocessing_classification_NLPLLM.ipynb`
**Phase 1 — EDA & Preprocessing**
- RAM-safe streaming via HuggingFace `datasets` (never loads all 5M rows)
- HTML/URL stripping, emoji→text conversion, near-duplicate removal (MD5 fingerprint)
- VADER sentiment scoring to detect star–sentiment mismatches (e.g. 2★ with positive text)
- Class balancing via stratified undersampling

**Phase 2 — Sentiment Classification**
- Model: `roberta-base` fine-tuned for 3-class sentiment
- Training innovations: weighted CrossEntropy + label smoothing (0.1), cosine LR schedule with warmup, gradient clipping, early stopping
- Evaluation: confusion matrix, ROC-AUC (OvR), calibration curves, per-class F1

### `Clustering_ProjectNLPLLM.ipynb`
- TF-IDF vectorisation + K-Means clustering to surface recurring product themes
- Silhouette analysis for optimal `k` selection
- Cluster labelling and export to `clustered_products.csv`

### `Summarization_Final.ipynb`
- Abstractive summarisation of review clusters using BART/T5
- Generates concise product insight cards from hundreds of reviews
- Chunking strategy for reviews exceeding model context length

---

##  Key Results

| Metric | Value |
|---|---|
| Sentiment classifier (RoBERTa) | ~91%+ accuracy |
| Macro F1 | >0.89 |
| Star–sentiment mismatch rate detected | ~14% of reviews |
| Clustering coherence (silhouette) | >0.42 |

> The **~14% mismatch rate** (high star + negative text, or low star + positive text) is a key business finding — star ratings alone are an unreliable proxy for true customer sentiment.

---

##  Quick Start

### Requirements
```bash
pip install transformers==4.40.1 datasets==2.19.1 accelerate==0.29.3 \
            scikit-learn==1.4.2 imbalanced-learn==0.12.2 \
            plotly==5.21.0 nltk textstat emoji wordcloud
```

### Recommended Runtime
- **Google Colab T4 GPU** (free tier is sufficient for Phases 1–2)
- Run notebooks in order: preprocessing → classification → clustering → summarisation

### Dataset
Loaded automatically via HuggingFace streaming — no manual download needed:
```python
from datasets import load_dataset
ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                  "raw_review_Electronics",
                  streaming=True, trust_remote_code=True)
```

---

##  Design Decisions

**Why RoBERTa over BERT/DistilBERT?**
RoBERTa removes the Next Sentence Prediction objective and uses dynamic masking with 10× more training data, which gives 2–4% accuracy improvements on review sentiment tasks at no extra inference cost.

**Why label smoothing?**
Amazon reviews with 3 stars are genuinely ambiguous — the same text could be positive or negative depending on context. Label smoothing (ε=0.1) prevents the model from becoming overconfident on these noisy boundary cases.

**Why streaming?**
Loading 5M reviews into a Colab DataFrame causes OOM crashes. Streaming yields rows one at a time, letting us collect exactly 120,000 samples without ever filling RAM.

**Why detect star–sentiment mismatches?**
Star ratings are set at click-time and often don't reflect review text (sarcasm, comparative statements, irrelevant complaints). VADER correlation analysis confirms ~14% of reviews have misaligned ratings, making raw star averages misleading for product teams.

---

## 📁 Output Files

| File | Description |
|---|---|
| `Reviews_classified_Roberta_VADER.csv` | All reviews with predicted sentiment label, VADER compound score, and star-sentiment agreement flag |
| `clustered_products.csv` | Products annotated with cluster ID and dominant topic keywords |

---

##  Demo App

Open `ReviewIntel_App.html` in any browser for an interactive dashboard that lets you explore classified reviews, filter by sentiment, and view cluster summaries — no server required.

---

##  Author

**Pavithra AI Engineer**
[github.com/pavithraaiengineer](https://github.com/pavithraaiengineer)

---

## 📄 License

This project is for educational and research purposes. The Amazon Reviews dataset is subject to its original licence terms at [McAuley Labs](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews).
