# flan_t5_train_fixed.py
import os, json, torch, numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)

# --- Config ---
DATA_PATH = "data/fee_messages_instruction.json"
MODEL_NAME = "google/flan-t5-small"
OUT_DIR = "./models/flan_t5_fee_extractor"
assert os.path.exists(DATA_PATH), "Dataset missing!"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load data ---
js = json.load(open(DATA_PATH))
ds = Dataset.from_dict({
    "input_text": [d["input_text"] for d in js],
    "target_text": [d["target_text"] for d in js],
})
split = ds.train_test_split(test_size=0.2, seed=42)
valid_test = split["test"].train_test_split(test_size=0.5, seed=42)
train_ds, val_ds = split["train"], valid_test["train"]

# --- Model & tokenizer ---
# Try loading tokenizer/model from the Hugging Face Hub; fall back to local OUT_DIR if offline
try:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Could not download tokenizer '{MODEL_NAME}': {e}\nFalling back to local '{OUT_DIR}'.")
    tok = AutoTokenizer.from_pretrained(OUT_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

try:
    # Do not move model to device here; Trainer will handle device placement.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Could not download model '{MODEL_NAME}': {e}\nFalling back to local '{OUT_DIR}'.")
    model = AutoModelForSeq2SeqLM.from_pretrained(OUT_DIR)


# --- Preprocess using new API ---
def preprocess(batch):
    # Tokenize inputs with padding (return plain python lists so datasets.Dataset stores them correctly)
    model_inputs = tok(
        batch["input_text"],
        max_length=256,
        padding="max_length",
        truncation=True,
    )

    # Tokenize targets (labels)
    labels = tok(
        batch["target_text"],
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    # Convert pad token ids in labels to -100 so they are ignored by the loss
    labels_ids = labels["input_ids"]
    masked_labels = [
        [(-100 if token_id == tok.pad_token_id else token_id) for token_id in label]
        for label in labels_ids
    ]

    # Return plain lists (datasets will convert to arrays) expected by Trainer/DataCollator
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": masked_labels,
    }

train_tok = train_ds.map(
    preprocess,
    batched=True,
    remove_columns=train_ds.column_names
)
val_tok = val_ds.map(
    preprocess, 
    batched=True,
    remove_columns=val_ds.column_names
)

print(train_tok)

# --- Data collator ---
collator = DataCollatorForSeq2Seq(
    tok,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

# --- Training args ---
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,   # keep simple to avoid accumulation issues
    learning_rate=5e-5,             # Reduced learning rate
    num_train_epochs=6,
    warmup_steps=100,               # Added warmup
    eval_steps=100,                 # More frequent evaluation
    save_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    weight_decay=0.01,
    fp16=False,  # keep mixed precision disabled to preserve stability
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_tok, eval_dataset=val_tok,
    tokenizer=tok, data_collator=collator
)

# --- Sanity check: disable fp16 for eval to avoid NaN ---
print("Running sanity eval (fp16 disabled)...")
orig_fp16 = args.fp16
args.fp16 = False
metrics = trainer.evaluate()
args.fp16 = orig_fp16

init_loss = metrics.get("eval_loss", float("nan"))
print(f"Initial eval_loss: {init_loss}")
if np.isnan(init_loss) or init_loss == 0:
    print("⚠️  Warning: Initial loss invalid. Will continue; training should fix it.")

# --- Train ---
trainer.train()
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)

# --- Quick inference ---
model.eval()
tests = [
    "Extract payment details: Hi Rahul Wells, we've received ₹12000 towards Exam Fee on September 12, 2025. Ref: TXN3211LGLI2411, Roll: R2052. - NextGen Learning Center",
    "Extract payment details: ₹10000 received from Michael Sharma (R4588) for Hostel Rent. Paid on March 04, 2025. Txn#: TXN6599OJYL2329. - Future Leaders College"
]
for t in tests:
    inputs = tok(t, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_length=128, num_beams=4)
    print("\nInput:", t[:80], "...\nOutput:", tok.decode(gen[0], skip_special_tokens=True))
