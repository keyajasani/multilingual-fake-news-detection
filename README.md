# Multilingual Fake News Detection Using XLM-RoBERTa

## Overview

This project investigates multilingual fake-news detection using
XLM-RoBERTa, a transformer-based language model.

The system classifies news articles as FAKE or REAL and provides
class probabilities. The project also explores model explainability
using SHAP and supporting-evidence retrieval from external sources.

Languages covered:
- English
- Hindi
- Gujarati

## Research Question

Can a multilingual transformer-based model accurately detect fake news
while providing explainable predictions, confidence scores, and
supporting evidence from trusted sources?

## Model

- Architecture: XLM-RoBERTa
- Base checkpoint: FacebookAI/xlm-roberta-base
- Task: Binary text classification
- Number of classes: 2
- Label 0: FAKE
- Label 1: REAL
- Training epochs: 2
- Learning rate: 2e-5

## Dataset

The research dataset contains 144,711 news articles.

| Language | Articles |
|---|---:|
| English | 62,702 |
| Hindi | 67,177 |
| Gujarati | 14,832 |
| Total | 144,711 |

The dataset was divided into training, validation, and test sets.

| Split | Articles |
|---|---:|
| Training | 115,767 |
| Validation | 14,471 |
| Test | 14,473 |

Dataset files and trained model weights are not included in this
repository. Confirm the dataset license and redistribution permissions
before sharing the data publicly.

## Evaluation Results

The saved test evaluation contains 14,473 examples.

| Metric | Result |
|---|---:|
| Accuracy | 99.53% |
| Precision (REAL positive) | 99.46% |
| Recall (REAL positive) | 99.76% |
| F1-score (REAL positive) | 99.61% |

The confusion matrix, evaluation reports, and language-wise results
are available in the `figures/`, `reports/`, and `results/` folders.

These results describe performance on the project's test set. They
should not be interpreted as a guarantee of performance on unseen
real-world news, other datasets, or all dialects and topics.

## Explainability

The project includes SHAP-based analysis of an example Hindi article.

Files are available in `explainability/`.

The example demonstrates a possible explanation workflow, not a
comprehensive validation of explanations across all languages or
articles. Explanations can also be affected by artifacts in the input.

## Evidence Retrieval

The project explores retrieving candidate supporting evidence for
selected articles.

See `evidence_retrieval/` for saved query plans and candidate records.

Important limitation: the exploratory baseline evaluated 10 articles
and produced 30 candidate records. Three candidates were flagged for
manual review, and no claims were automatically verified.

A retrieved URL or a high semantic similarity score does not prove
that a source supports a claim. The evidence-retrieval component should
therefore be treated as exploratory and requires human verification.

## Repository Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── reports/
├── results/
├── explainability/
├── evidence_retrieval/
└── figures/
```
