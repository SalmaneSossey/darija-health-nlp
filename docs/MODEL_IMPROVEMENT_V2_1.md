# V2.1 Darija Model Improvement Sprint

## Goal

Improve the Darija Health NLP specialty model with a reproducible workflow that can be defended in front of a technical supervisor:

- no train/test leakage;
- explicit dataset provenance;
- GPU training on Kaggle;
- CPU deployment on the Pulsaride VPS;
- honest before/after metrics from the integrated Pulsaride endpoint.

## Current Baseline

The current deployed Pulsaride V2 integration uses the Darija Health NLP MARBERT artifact through `POST /ai/triage`.

Latest integrated evaluation:

- API success rate: `100%`
- Specialty accuracy: `55.82%`
- Specialty macro F1: `47.87%`
- Red-flag recall: `100%`
- P95 latency: about `603 ms`

This is acceptable for a guarded demo, but not strong enough to claim a mature medical specialty classifier.

## Dataset Provenance

Processed splits must exist before training:

```text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
```

If they are missing, restore the raw MedQA-MA dataset under:

```text
data/raw/medqa_ma/
```

Then rebuild:

```bash
python src/data/build_processed_dataset.py
python src/data/write_dataset_manifest.py
```

The manifest records row counts, label distributions, source distributions, hashes, and leakage checks. Raw and processed data stay out of Git unless the dataset license allows publishing them.

## Kaggle Training

Create a Kaggle package:

```bash
python src/data/package_kaggle_training.py
```

If the processed splits are not on the local machine yet, create a code-only handoff package:

```bash
python src/data/package_kaggle_training.py --allow-missing-data
```

On Kaggle, enable GPU and run:

```bash
python src/models/train_transformer_specialty_classifier.py \
  --model marbert \
  --epochs 4 \
  --batch-size 16 \
  --max-length 160
```

Then compare with class weighting:

```bash
python src/models/train_transformer_specialty_classifier.py \
  --model marbert \
  --epochs 4 \
  --batch-size 16 \
  --max-length 160 \
  --class-weighted
```

Optional comparison, if GPU time remains:

```bash
python src/models/train_transformer_specialty_classifier.py --model arabert --epochs 4 --batch-size 16 --max-length 160
python src/models/train_transformer_specialty_classifier.py --model mbert --epochs 4 --batch-size 16 --max-length 160
```

Use `valid.csv` for model selection. Use `test.csv` only for the final result.

## Expected Export

The selected model folder must contain:

```text
model.safetensors
config.json
tokenizer.json
tokenizer_config.json
training_manifest.json
test_metrics.json
```

Copy that folder to:

```text
models/transformer_MARBERT_specialty/
```

Then redeploy the Darija AI service and rerun the Pulsaride integrated evaluator.

## Interpretation Rules

- The transformer predicts the specialty.
- Deterministic safety rules protect red-flag urgency.
- OpenAI is not used for this sprint because it is paid and would weaken the local deployability story.
- The objective is software/AI engineering validation, not medical validation.
