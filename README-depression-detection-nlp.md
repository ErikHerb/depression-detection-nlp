# Depression Detection from Clinical Interview Transcripts

Predicting PHQ-8 depression-severity from the words people use in clinical interviews, and comparing classical machine learning against a fine-tuned BERT transformer.

## Overview

Depression is often underdiagnosed, and its symptoms surface in how people talk. This project tests whether text alone — transcripts of clinical interviews — can flag depression risk, which could help screening scale beyond the clinic. Each team member framed the task as a binary classification at a different PHQ-8 cutoff, then compared eight classical models against a transformer-based model.

**Project question:** *Can we predict a participant's PHQ-8 severity range from the words in their interview?*

## Dataset

- **Source:** DAIC-WOZ (USC ICT Distress Analysis Interview Corpus) — semi-structured interviews conducted by a virtual interviewer, each paired with a transcript and a PHQ-8 score
- **Size:** 141 participants after cleaning
- **Modality:** transcript text only
- **My band (Erik):** PHQ-8 **5–9 (mild)**, a 75/25 class split

> **Important:** DAIC-WOZ is access-restricted clinical data. **No transcripts, scores, or any participant data are included in this repository** — only the analysis code. Access must be obtained directly from USC ICT under their data-use agreement.

## Approach

The project is split across four notebooks:

| Notebook | Purpose |
|---|---|
| `Project_2_Proposal_NLP.ipynb` | Load the DAIC-WOZ subset (ids, transcripts, labels) and define the task |
| `NLP_P2_N1_Feature_Sets.ipynb` | Build feature sets: TF-IDF (Porter stemming), TextBlob sentiment/subjectivity, and PCA |
| `NLP_P2_N2_Classification.ipynb` | Train and evaluate eight classical models across repeated random splits |
| `NLP_P2_N3_Bert.ipynb` | Fine-tune a BERT transformer with a learning-rate sweep |

- **Classical track:** Random Forest, Naive Bayes, kNN, Logistic Regression, two SVMs, AdaBoost, and XGBoost on TF-IDF + PCA features, evaluated over 15 runs with balanced accuracy, sensitivity, and specificity.
- **Transformer track:** pretrained `bert-base-uncased` with a frozen encoder and a small MLP head, class-weighted loss, and a learning-rate sweep, capped at 512 tokens.

## Key Results

- **Best overall result (my mild band): Logistic Regression, 0.63 balanced accuracy** (sensitivity 0.59 / specificity 0.66) — the strongest result across the team.
- Regularized linear models led whenever classes were closer to balanced; extreme severity bands hovered near chance because shallow text features couldn't separate them.
- **BERT did not beat the simple baseline.** With ~98 training examples and heavy class imbalance, the frozen-encoder model often collapsed to predicting the majority class (balanced accuracy ≈ 0.50).
- **Takeaway:** with small, imbalanced clinical data, well-regularized classical models can outperform a large transformer — a practical lesson in matching model complexity to data size.

## Results in Detail

All figures are 15-run averages on the mild (PHQ-8 5–9) band. The aggregate
numbers and the script that produces these charts are in [`results/`](results/) —
no participant data is included.

**Classical models (mild band):**

| Model | Balanced Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| **Logistic Regression** | **0.63 ± 0.09** | 0.59 | 0.66 |
| SVM (RBF) | 0.62 ± 0.08 | 0.62 | 0.61 |
| SVM (linear) | 0.61 ± 0.07 | 0.63 | 0.58 |
| AdaBoost | 0.58 ± 0.10 | 0.47 | 0.70 |
| XGBoost | 0.57 ± 0.08 | 0.36 | 0.78 |
| Random Forest | 0.56 ± 0.08 | 0.43 | 0.70 |
| Naive Bayes | 0.54 ± 0.06 | 0.75 | 0.32 |
| kNN | 0.53 ± 0.12 | 0.52 | 0.54 |

![Balanced accuracy by classical model](results/balanced_accuracy_by_model.png)

**Classical baseline vs. fine-tuned BERT:** the best classical model reached 0.63
balanced accuracy, while the fine-tuned BERT settled at 0.45–0.48 — below chance —
collapsing to majority-class predictions under the small, imbalanced data.

![Best classical model vs BERT](results/classical_vs_bert.png)

## Tech Stack

Python · scikit-learn · XGBoost · NLTK · TextBlob · TF-IDF · PCA · PyTorch · Hugging Face Transformers (BERT)

## How to Run

```bash
pip install scikit-learn xgboost nltk textblob pandas numpy seaborn matplotlib torch transformers
```

Obtain DAIC-WOZ access, place your data subset where the proposal notebook expects it, then run the notebooks in order: proposal → feature sets → classification → BERT. A GPU is recommended for the BERT notebook.

## Team & Role

Group project for **MSDS 620 — Natural Language Processing**. Team: Simon Salaj, Grant Robinson, Jackson Swallow, Erik Herb. I owned the **mild (PHQ-8 5–9)** band and its feature, classification, and evaluation work.

## Limitations & Next Steps

- Small dataset and class imbalance limit deep-model performance.
- Text-only — audio cues like pauses and prosody were excluded.
- Next steps: train BERT on the full PHQ-8 regression instead of per-band binary cutoffs, unfreeze the encoder with early stopping, add acoustic features, and use k-fold cross-validation for tighter variance estimates.
