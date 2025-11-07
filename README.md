# Fee Receipt Extraction & PDF Generator (FLAN-T5 fine-tuned)

Professional, end-to-end example project that demonstrates generating a synthetic dataset of payment messages, fine-tuning a FLAN-T5 Seq2Seq model to extract structured payment details, and serving a Streamlit app that converts free-form payment messages into an official PDF receipt.

## Highlights
- Synthetic dataset generator (human-like payment messages): `dataset.py`
 - Synthetic dataset generator (human-like payment messages): `utils/dataset.py`
- Training script using Hugging Face Transformers + Datasets: `train_model.py`
- Streamlit web app that performs inference and generates printable PDF receipts: `app.py`
- Model artifacts live in `models/flan_t5_fee_extractor/` (tokenizer + safetensors compatible model)

## Repository layout

```
.
├─ app.py                  # Streamlit app for inference + PDF receipt generation
├─ train_model.py          # Training script using HF Trainer
├─ utils/                  # Utility scripts
│  ├─ dataset.py           # Synthetic dataset generation (writes to data/fee_messages_instruction.json)
│  └─ eval_dataset.py      # Evaluation dataset helpers
├─ data/                   # Generated dataset (ignored by .gitignore)
└─ models/                 # Fine-tuned model artifacts (ignored by .gitignore)
```

## Quick setup (Windows / PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Upgrade pip then install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Install a GPU build of `torch` if you have CUDA-enabled hardware. See https://pytorch.org for the correct command for your CUDA version.
- If you get model-loading errors due to missing optional packages (e.g., `safetensors`), install them with `pip install safetensors`.

## Generate the dataset

The repository includes `utils/dataset.py` which generates a synthetic dataset of payment messages and writes `data/fee_messages_instruction.json`.

```powershell
python utils/dataset.py
```

This creates ~2000 training samples by default. You can edit `dataset.py` to change sample count or templates.

## Training the model

Use `train_model.py` to fine-tune a FLAN-T5 model (the script will attempt to download `google/flan-t5-small` and fall back to `models/flan_t5_fee_extractor` if offline).

Important notes:
- Training benefits from a GPU. If you train on CPU it will be slow.
- Check `train_model.py` training hyperparameters and adjust `per_device_train_batch_size`, `num_train_epochs`, and `learning_rate` as needed.

Run training:

```powershell
python train_model.py
```

After training completes, the model and tokenizer are saved to `./models/flan_t5_fee_extractor`.

## Running the Streamlit app (inference + PDF generation)

Start the app:

```powershell
streamlit run app.py
```

App features:
- Paste a payment SMS/email text or select a sample message.
- The app calls the fine-tuned FLAN-T5 model to extract fields like student_name, amount, roll_number, date, transaction_id, payment_name, and institute.
- Generates a professional PDF receipt (ReportLab) which you can download.

## Model & Tokenizer

The project uses Hugging Face Transformers. A small pre-trained model name used in the scripts is `google/flan-t5-small`. The repo also includes a `models/flan_t5_fee_extractor` folder — if present, the scripts will prefer that offline fallback.

If you re-train the model, save the artifacts and commit only lightweight pointers (do not commit large model files) — the `.gitignore` excludes `models/` and `data/` by default.

## Model evaluation summary

I ran an evaluation of the fine-tuned model and saved the full results to `metrics/metrics_summary.json` (inside the `metrics/` folder). Key highlights (1000 eval samples):

| Metric | Value |
|---|---:|
| Samples evaluated | 1000 |
| Exact match | 23.1% |
| Normalized exact match | 23.1% |
| ROUGE-L (precision) | 0.8813 |
| ROUGE-L (recall) | 0.9409 |
| ROUGE-L (F1) | 0.8978 |
| Latency mean (s / item) | 0.22596 |
| Latency p50 (s) | 0.24202 |
| Latency p95 (s) | 0.26678 |
| Throughput (items / s) | 4.4256 |

Per-field extraction metrics (precision / recall / F1):

| Field | Precision | Recall | F1 | TP |
|---|---:|---:|---:|---:|
| student_name | 1.000 | 1.000 | 1.000 | 1000 |
| transaction_id | 0.995 | 0.952 | 0.973 | 952 |
| payment_name | 0.996 | 0.995 | 0.995 | 995 |
| institute | 0.862 | 0.847 | 0.854 | 847 |
| date | 1.000 | 0.957 | 0.978 | 957 |
| roll_number | 0.511 | 0.511 | 0.511 | 511 |
| amount | 0.473 | 0.473 | 0.473 | 473 |

Notes and interpretation:

- High performance on textual fields like `student_name`, `transaction_id`, and `payment_name` indicates the model reliably copies or formats those values from messages.
- Lower scores on `amount` and `roll_number` suggest the model sometimes changes formatting (currency tokens, prefixes like `RN`/`Roll_`) or omits/duplicates values in noisy templates — consider normalizing numeric formats in preprocessing and/or adding more diverse examples.
- `date` and `institute` perform well but have a few misses — augmenting the training set with more date formats and institute name variations could help.

See `metrics/metrics_summary.json` for the full predictions and the raw metrics object.

## Troubleshooting

- Model loading fails with OOM: reduce `per_device_train_batch_size` or use gradient accumulation. Consider training on a machine with more GPU RAM.
- `torch` not found or CUDA mismatch: reinstall `torch` with the matching CUDA support for your system.
- Streamlit caching/loading issues: restart the Streamlit server when you update model files.

## License & Acknowledgements

This repository is provided as an example/demo. Review any third-party model or dataset license terms before using in production.


