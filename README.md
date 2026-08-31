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
- Optional native Rust inference engine (PyO3) for the classical SVM, with transparent joblib fallback.
- Docker Compose for local deployment.

## Architecture

```text
data/raw/medqa_ma -> EDA -> data/processed -> TF-IDF classifier (joblib)
                                              |  -> export_svm_to_rust.py
                                              |       -> models/svm_weights.json
                                              |       -> rust_inference (PyO3, cdylib)
                                      -> FastAPI /predict  (Transformer -> Rust SVM -> joblib)
                                      -> Streamlit UI
```

Reusable data, feature, and model code lives in `src/`. The native Rust inference engine lives in `rust_inference/` (compiled as a Python extension via PyO3). Application code lives in `backend/` and `frontend/`. Generated data, models, and artifacts are ignored by Git.

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

For the V2.1 model-improvement sprint, generate dataset provenance before training:

```bash
python src/data/write_dataset_manifest.py
```

Class-weighted MARBERT can be trained on Kaggle GPU:

```bash
python src/models/train_transformer_specialty_classifier.py \
  --model marbert \
  --epochs 4 \
  --batch-size 16 \
  --max-length 160 \
  --class-weighted
```

Create a Kaggle-ready package with:

```bash
python src/data/package_kaggle_training.py
```

See `docs/MODEL_IMPROVEMENT_V2_1.md` for the full workflow.

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

## V3: native Rust inference engine (PyO3)

The V2 classical SVM is exported to a flat JSON weight file and re-implemented in Rust as a PyO3 extension module (`rust_inference/`). The backend loads it transparently, eliminating the scikit-learn import on the inference path.

### Three-tier fallback pipeline

The backend picks the highest-priority backend that is available at startup:

- **Tier 1 — Transformer (MARBERT):** loaded when `models/transformer_MARBERT_specialty/` exists and `transformers` is importable.
- **Tier 2 — Native Rust SVM:** loaded when the compiled `rust_inference` extension is importable and `models/svm_weights.json` exists. No `scikit-learn` is loaded at runtime.
- **Tier 3 — Joblib scikit-learn:** loaded only if neither Tier 1 nor Tier 2 is available.

### Build the Rust extension

```bash
python -m pip install maturin
cd rust_inference
maturin develop --release
```

Requires a Rust toolchain (`cargo`, `rustc`) and a Python version supported by the pinned PyO3 (this project uses Python 3.13 and PyO3 0.22).

### Export the V2 weights to JSON

```bash
python src/models/export_svm_to_rust.py
```

Output:

- `models/svm_weights.json` (vocabulary, IDF, class list, weight matrix, intercepts)

### V3 results

The Rust engine re-implements the V2 `char_wb` TF-IDF + LinearSVC pipeline: `char_wb` 3-5 n-gram extraction, L2 normalization, and multiclass LinearSVC argmax. Re-implementation parity with scikit-learn is exact.

#### Held-out test set (n = 9965, 25 classes)

All three backends scored on the same split (`data/processed/test.csv`).

| Backend                       | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | µs / call |
| ----------------------------- | -------- | -------- | ----------- | --------------- | ------------ | --------- |
| **MARBERT** (Transformer, CPU fp32) | `0.6845` | `0.6755` | `0.6836`    | `0.6822`        | `0.6781`     | `21 609.5` |
| **Rust SVM** (PyO3, native)         | `0.6492` | `0.6255` | `0.6442`    | `0.6178`        | `0.6396`     | `21.8`     |
| **sklearn LinearSVC** (joblib)      | `0.6492` | `0.6255` | `0.6442`    | `0.6178`        | `0.6396`     | `62.4`     |

The full per-class report, confusion matrix, and combined metrics are in `artifacts/metrics/comparison_all_tiers.json` and `artifacts/metrics/specialty_metrics.json`.

#### Accuracy vs latency trade-off

```
Macro F1
  0.68 |                          * MARBERT
  0.67 |
  0.66 |
  0.65 |
  0.64 |
  0.63 |
  0.62 |  * Rust SVM
  0.62 |  * sklearn LinearSVC
       +--------------------------------------------
         0      20k      40k      60k     µs / call
```

MARBERT buys +5.0 pp macro-F1 (and +3.5 pp accuracy) over the classical SVM, at roughly **1000× the per-call latency on CPU**. For a latency-sensitive deployment, the Rust SVM tier is the default; the transformer is a Tier-1 opt-in for high-accuracy offline scoring.

#### Agreement and determinism

- Rust vs sklearn agreement on the full test set: **9965 / 9965 (100.00%)** — predictions, scores, and argmax indices are bit-identical to the scikit-learn pipeline.
- Per-call latency (isolated micro-benchmark, single warm input):

  | Backend       | Latency   | Speedup |
  | ------------- | --------- | ------- |
  | scikit-learn  | `347.3 µs` | 1.0×   |
  | Rust (PyO3)   | `6.7 µs`   | `51.8×` |

  The 51.8× number is the most relevant figure for a request-driven FastAPI service: each `/predict` call hits a warm Rust object. Full-batch (n = 9965) throughput speedup is `2.9×` because Python list-comprehension overhead and tokenizer step-up dominate the batch loop, not the kernel.

Notes:

- MARBERT timing above is a CPU re-run (PyTorch fp32). The training-time figure captured in `artifacts/metrics/transformer_MARBERT_local_specialty_metrics.json` (`eval_runtime: 24.6 s`, `404.6 samples/s`) was measured on the same GPU/FP16 configuration the model was trained on; accuracy and F1 are identical between the two runs (`0.6845` / `0.6755`).
- The JSON weight file is ~80 MB; for production a binary format (`bincode` or raw `f32` little-endian) would cut load time and disk footprint significantly.
- Transformer dependencies are still intentionally not part of the Rust extension; the optional MARBERT path remains an opt-in.

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
