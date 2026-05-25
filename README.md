# Darija Health NLP

Moroccan Medical Triage System using NLP, FastAPI, Streamlit, Docker, and Colab.

This project studies whether NLP models can classify Moroccan patient messages written in Darija, Arabic script, French, or mixed language into relevant medical specialties and urgency levels.

## Medical disclaimer

This project is for academic and educational purposes only. It does not provide medical diagnosis, treatment, or emergency medical advice. Users should consult qualified healthcare professionals for medical concerns.

## Problem statement

Moroccan patients often describe symptoms in Moroccan Darija, Latin-script Arabizi, Arabic script, French, or a mixture of these. Standard medical NLP pipelines are usually not designed for this linguistic setting. This project builds a practical V1 pipeline for orientation: specialty classification, rule-based urgency detection, multilingual symptom extraction, and safe recommendations.

## Features

- Dataset inspection and EDA for MedQA-MA.
- Light Darija/Arabic/French text normalization.
- Unified processed schema: `id,text,language,specialty,urgency,symptoms,source`.
- TF-IDF baselines with Logistic Regression and Linear SVM.
- Dictionary-based symptom extraction.
- Rule-based urgency orientation.
- FastAPI backend and Streamlit frontend.
- Docker Compose for local deployment.

## Architecture

```text
data/raw/medqa_ma -> EDA -> data/processed -> TF-IDF classifier
                                      -> FastAPI /predict
                                      -> Streamlit UI
```

Reusable data, feature, and model code lives in `src/`. Application code lives in `backend/` and `frontend/`. Generated data, models, and artifacts are ignored by Git.

## Dataset

The project uses MedQA-MA. Copy it from Windows into WSL:

```bash
mkdir -p data/raw/medqa_ma
cp -r "/mnt/c/Users/lione/Downloads/MedQA-MA; Question Answering Dataset in Moroccan A/MedQA-MA; Question Answering Dataset in Moroccan A/"* data/raw/medqa_ma/
find data/raw/medqa_ma -maxdepth 3 -type f | head -50
```

The current inspected dataset includes a canonical master CSV at:

```text
data/raw/medqa_ma/Dataset/MedQA_Ma dataset/MedQA_MA.csv
```

with columns `Question`, `Answer`, and `Category`.

## EDA

Run:

```bash
python src/data/inspect_dataset.py
python src/data/run_eda.py
```

EDA outputs:

- `artifacts/figures/class_distribution.png`
- `artifacts/figures/text_length_distribution.png`
- `artifacts/figures/top_tokens.png`
- `artifacts/figures/missing_values.png`
- `artifacts/figures/language_distribution.png`
- `artifacts/reports/eda_summary.md`

The initial EDA found 100,966 rows, 24 categories, mostly Arabic-script Darija, repeated questions, and some long/noisy text. These findings justify light normalization, duplicate removal, and a custom triage supplement for urgency and symptoms.

## Preprocessing

Run:

```bash
python src/data/create_custom_triage_examples.py
python src/data/build_processed_dataset.py
```

The preprocessing keeps Darija Arabizi digits such as `3`, `7`, and `9`, normalizes Arabic letter variants, removes excessive whitespace and punctuation, removes empty text rows, and deduplicates exact text-specialty pairs.

The custom triage generator creates 500+ safe orientation examples, with extra Latin Darija and French/Darija mixed cases for urgency and symptom extraction.

Outputs:

- `data/processed/triage_dataset.csv`
- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/processed/test.csv`
- `data/sample/sample_medqa_ma.csv`

## Model training

Run:

```bash
python src/models/train_specialty_classifier.py
```

The training script compares:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Character `char_wb` TF-IDF + Linear SVM

The best validation macro F1 model is saved to `models/specialty_classifier.joblib`.

V2 adds character n-grams because Darija, Arabizi, Arabic, and French spelling can vary heavily:

```python
TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
```

You can also train with broader labels or merge rare labels:

```bash
python src/models/train_specialty_classifier.py --label-mode broad
python src/models/train_specialty_classifier.py --label-mode rare_merged --min-class-count 50
```

The default keeps the original specialty labels so the V1 API contract remains stable.

## Optional transformer comparison

Keep TF-IDF + LinearSVC as the baseline. For comparison experiments with MARBERT, AraBERT, or multilingual BERT, install optional dependencies:

```bash
pip install -r requirements-transformers.txt
```

Then run one of:

```bash
python src/models/train_transformer_specialty_classifier.py --model marbert
python src/models/train_transformer_specialty_classifier.py --model arabert
python src/models/train_transformer_specialty_classifier.py --model mbert
```

For Colab, use:

```text
notebooks/07_transformer_comparison.ipynb
```

Open directly in the browser:

[Open 07 Transformer Comparison in Colab](https://colab.research.google.com/github/SalmaneSossey/darija-health-nlp/blob/main/notebooks/07_transformer_comparison.ipynb)

Colab runs on a remote VM and cannot access local WSL files. The notebook can clone this GitHub repository automatically, but you must still provide data because raw data, processed data, models, and artifacts are intentionally ignored by Git. The easiest option is to upload or copy:

```text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
```

into the Colab repo before starting transformer training. To make upload easier, create a Colab-ready zip locally:

```bash
python src/data/export_colab_training_data.py
```

Then upload this file in the transformer notebook:

```text
artifacts/colab/darija_health_processed_splits.zip
```

The notebook extracts it into `data/processed/`. Alternatively, upload the raw MedQA-MA dataset to `data/raw/medqa_ma/` in Colab and rerun the preprocessing pipeline.

Transformer dependencies are intentionally not included in the backend or frontend Docker images.

## Evaluation

Run:

```bash
python src/models/evaluate_model.py
python src/models/analyze_v2_errors.py
```

Outputs:

- `artifacts/metrics/specialty_metrics.json`
- `artifacts/figures/confusion_matrix.png`
- `artifacts/reports/error_analysis.md`

## V2 results

The best V2 classical baseline is `char_wb_tfidf_linear_svm`, using character n-grams with `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))`.

Validation comparison:

- TF-IDF Logistic Regression: macro F1 `0.6163`
- Word TF-IDF LinearSVC: macro F1 `0.6198`
- Character TF-IDF LinearSVC: macro F1 `0.6281`

Held-out test performance for the selected V2 model:

- Accuracy: `0.6492`
- Macro F1: `0.6255`
- Weighted F1: `0.6442`

The main confusion patterns are clinically and lexically plausible: Anesthesiology vs Neurology, Cardiology vs Hematology, Dermatology vs Cosmetic Dermatology, and General Practice/Internal Medicine-style labels. Error analysis also shows that very short questions have the highest error rate, while Latin Darija and French scores are optimistic because those examples mostly come from the small custom set rather than real MedQA-MA messages.

Detailed V2 error-analysis outputs are generated by:

```bash
python src/models/analyze_v2_errors.py
```

Key artifacts:

- `artifacts/reports/v2_error_analysis.md`
- `artifacts/metrics/v2_top_confusions.csv`
- `artifacts/metrics/v2_language_error_analysis.csv`
- `artifacts/metrics/v2_per_class_metrics.csv`
- `artifacts/figures/v2_normalized_confusion_matrix.png`
- `artifacts/figures/v2_top_confusions.png`
- `artifacts/figures/v2_language_error_rates.png`

## Backend

Run locally:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "3ndi wje3 f sedri w di9 f nefs"}'
```

## Frontend

Run locally:

```bash
cd frontend
API_URL=http://localhost:8000/predict streamlit run app.py
```

## Docker

Run:

```bash
docker compose up --build
```

Expected access:

- Frontend: `http://localhost:8501`
- Backend docs: `http://localhost:8000/docs`

## Colab notebooks

Use the notebooks in order:

1. `01_data_inspection.ipynb`
2. `02_eda.ipynb`
3. `03_preprocessing_pipeline.ipynb`
4. `04_train_specialty_classifier.ipynb`
5. `05_urgency_rules_and_symptom_extraction.ipynb`
6. `06_evaluation_and_error_analysis.ipynb`
7. `07_transformer_comparison.ipynb`

Each notebook calls reusable code from `src/` or `backend/`.

## Limitations

- The deployed specialty model is a classical TF-IDF baseline; transformer comparison is available as an optional Colab experiment.
- Urgency is rule-based and conservative.
- Symptom extraction is dictionary-based and incomplete.
- MedQA-MA contains vague, noisy, and assistant-like rows.
- Latin-script Darija and French are underrepresented in the raw dataset, so custom examples are included for V1 behavior.

## Ethical considerations

The system must never diagnose. It provides orientation only, uses high-urgency safety recommendations, and always displays a disclaimer. For emergency-like messages, users are told to seek urgent professional care.

## Future work

- Better Darija normalization and spelling variation handling.
- Larger clinically reviewed custom triage examples.
- ML-based urgency classifier.
- Transformer baselines after the classical model is established.
- MLflow tracking.
- Retrieval over vetted Moroccan public health guidance.
- Deployment hardening and monitoring.
