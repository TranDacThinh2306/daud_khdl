# Depression Alert System

An Explainable AI supported Alert system for detecting depression indicators in social media comments. This project combines NLP-based feature extraction, DistilBERT and explainability techniques (SHAP, LIME) to provide transparent and interpretable predictions.

## Overview

This system analyzes social media comments to identify potential depression indicators using:
- **NLP Feature Extraction**: TF-IDF, BoW, N-grams, GloVe embeddings
- **Linguistic Analysis**: LIWC-style linguistic markers
- **Behavioral Features**: Screen time patterns, nighttime usage
- **Multiple ML Models**: Random Forest, SVM, XGBoost, LSTM, BERT
- **Explainability**: SHAP and LIME for transparent predictions

## Project Structure

```
depression_xai_system/
├── data/                    # Raw, processed, and external datasets
├── notebooks/               # Jupyter notebooks for EDA and prototyping
├── src/                     # Source code
│   ├── data/                # Data collection and preprocessing
│   ├── features/            # Feature engineering (NLP focus)
│   ├── models/              # Classification models
│   ├── explainability/      # SHAP, LIME, and visualization
│   ├── pipelines/           # End-to-end pipeline orchestration
│   ├── api/                 # REST API for inference
│   ├── utils/               # Utility functions
│   └── config/              # Configuration files
├── models_saved/            # Saved model artifacts
├── reports/                 # reports and figures
└── scripts/                 # Utility scripts
```

## Quick Start

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yaml
conda activate depression_xai
```

### Running Scripts

You can run the main functionalities using the Python scripts provided in the `scripts/` directory:

```bash
# Run model training and generate XAI explanations
python -m scripts.run_explain --model_path models_saved/experiments --num_samples 50 --output_dir reports/figures
python -m scripts.run_explain --model_path models_saved/experiments --num_samples all --output_dir reports/figures
# Note: num_samples all will take a lot of time


# Crawl data using proxy rotation
python -m scripts.crawl_with_proxy_rot --subreddits depression anxiety --limit 500 --proxy

# Scrape additional dataset with rotation config
python -m scripts.data_scraper --subreddits mentalhealth --limit 300 --rotation 50 --delay 2.0
```

### API

```bash
# Start the API server
make api

# Or using Docker
docker-compose up
```

### Endpoints

| Endpoint     | Method | Description                          |
|-------------|--------|--------------------------------------|
| `/predict`  | POST   | Predict depression indicators        |
| `/explain`  | POST   | Get prediction with SHAP/LIME explanation |
| `/batch`    | POST   | Batch prediction for multiple comments |

## Configuration

Configuration files are in `src/config/`:
- `config.yaml` — General system configuration
- `model_config.yaml` — Model hyperparameters
- `xai_config.yaml` — SHAP/LIME settings

## References

- eRisk Dataset for depression detection
- SMHD (Self-reported Mental Health Diagnoses)
- LIWC (Linguistic Inquiry and Word Count)

## License

This project is for research purposes only.
