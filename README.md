# Duplicate Question Detection on Quora

A deep learning project that compares a **Siamese LSTM** and a **fine-tuned BERT** model for duplicate question detection on the Quora Question Pairs dataset. The project also uses **LIME** and **SHAP** to explain token-level model predictions and analyze how recurrent and transformer-based architectures differ in their decision-making.

## Overview

This project was developed to study semantic similarity between question pairs and evaluate how well different neural architectures can identify duplicates in a real-world NLP task. It combines model implementation, training, evaluation, and interpretability in one workflow, making it useful both as a machine learning project and as a comparative study of sequence models versus transformers.

## Features

- Duplicate question detection using the Quora Question Pairs dataset.
- Comparison of two architectures: Siamese LSTM and BERT.
- End-to-end preprocessing, training, validation, and evaluation pipeline.
- Interpretability analysis with LIME and SHAP for token-level explanations.
- Qualitative comparison of how both models respond to duplicate and non-duplicate examples.

## Dataset

The project uses the **Quora Question Pairs** dataset with 404,291 labeled question pairs, split into training and validation using an 80/20 stratified setup, plus a small 20-pair test file for qualitative checks. The task is binary classification: predict whether two questions are duplicates or not.

| Split | Number of pairs |
|-------|-----------------|
| Training | 323,433 |
| Validation | 80,858 |
| Testing | 20 |

## Models

### Siamese LSTM

The Siamese LSTM encodes each question separately using pre-trained Google News word2vec embeddings with 300 dimensions, followed by an LSTM encoder with hidden size 50. The two encoded vectors are compared using Manhattan distance to estimate semantic similarity between the question pair.

Key details:
- Pre-trained word2vec embeddings (Google News, 300d).
- LSTM hidden size of 50.
- Two-phase training schedule with frozen embeddings first, then embedding fine-tuning after 30 epochs.
- Mean squared error loss on similarity scores.

### BERT

The transformer baseline uses **bert-base-uncased** for direct pair classification in the format `[CLS] Q1 [SEP] Q2 [SEP]`.[file:2] The final hidden state of the `[CLS]` token is passed to a linear classifier to predict duplicate vs. not duplicate.

Key details:
- bert-base-uncased tokenizer and encoder.
- Cross-entropy loss with AdamW optimization.
- Training batch size of 32 and validation/test batch size of 64.
- Converged within 2 epochs in the reported setup.

## Training setup

Experiments were run in **Google Colab** with GPU acceleration when available. The Siamese LSTM uses Adam with learning rate `1e-3`, weight decay `1e-5`, batch size 128, and up to 60 epochs, while BERT is fine-tuned with AdamW at `2e-5` for 2 epochs.

Preprocessing differed by model:
- Siamese LSTM: lowercasing, regex cleaning, stopword removal, vocabulary building, index mapping, and left-padding to 40 tokens.
- BERT: standard BERT tokenization with truncation/padding to a maximum sequence length of 64.

## Results

BERT achieved stronger validation performance than the Siamese LSTM in this study, reaching 90.5% validation accuracy and 0.874 F1, compared with 84.2% validation accuracy for the Siamese LSTM.

| Model | Validation Accuracy | F1 Score |
|-------|---------------------|----------|
| Siamese LSTM | 84.2%  | — |
| BERT | 90.5% | 0.874 |

## Interpretability

LIME and SHAP were applied to both models to explain predictions at the token level.[file:2] The analysis showed that the Siamese LSTM distributes importance more broadly across shared and mismatched terms, while BERT places stronger emphasis on highly discriminative entity tokens such as country names or named entities.

Example findings:
- In non-duplicate pairs, entity mismatches such as `china` vs. `korea` strongly pushed predictions toward **Not Duplicate**.
- In duplicate pairs, entity tokens such as `chinese` and `trump` strongly supported the **Duplicate** prediction.
- BERT produced sharper and more localized attribution patterns than the Siamese LSTM.

## Tech stack

- Python
- PyTorch
- BERT / Transformers
- Siamese LSTM 
- LIME 
- SHAP 
- Google Colab 
- NLP / Text Classification 

## Conclusion

This project highlights the trade-off between lightweight recurrent architectures and larger transformer-based models for semantic similarity tasks. While the Siamese LSTM offers a simpler similarity-based setup, BERT delivers stronger performance and sharper semantic discrimination on duplicate question detection.
