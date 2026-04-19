

import pandas as pd
import torch
import numpy as np
import evaluate
import re
from datasets import Dataset
from transformers import (
    MT5ForConditionalGeneration, 
    MT5Tokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 2. LOAD DATASET 
# ==========================================

def load_dataset_from_colab(file_path):
    df = pd.read_csv(file_path).astype(str)
    
    full_dataset = Dataset.from_pandas(df)
    return full_dataset.train_test_split(test_size=0.1)

# ==========================================
# 3. TOKENIZATION & PREPROCESSING
# ==========================================
def preprocess_function(examples, tokenizer):
    
    inputs = examples["input"]
    targets = examples["output"]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ==========================================
# 4. MODEL & LORA SETUP
# ==========================================
def setup_model():
    model_name = "google/mt5-small"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # Applying LoRA hyperparameters from Methodology 
    lora_config = LoraConfig(
        r=16,                # Rank: 16 [cite: 50]
        lora_alpha=32,       # Alpha: 32 [cite: 51]
        target_modules=["q", "v"], # Target Query and Value matrices 
        lora_dropout=0.05,
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

# ==========================================
# 5. MULTI-ANGLE EVALUATION (BLEU & chrF)
# ==========================================
# Using the multi-angle approach to capture Amharic linguistic depth [cite: 59, 61]
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]

    return {
        "bleu": bleu.compute(predictions=decoded_preds, references=decoded_labels)["score"],
        "chrf": chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    }

# ==========================================
# 6. TRAINING EXECUTION
# ==========================================
# Initialize Data and Model
raw_data = load_dataset_from_colab("cleaned_dataset.csv")
model, tokenizer = setup_model()

# Process Dataset
tokenized_data = raw_data.map(
    lambda x: preprocess_function(x, tokenizer), 
    batched=True
)

# Set Training Arguments for T4 GPU [cite: 48]
training_args = Seq2SeqTrainingArguments(
    output_dir="./amharic-en-results",
    evaluation_strategy="epoch",
    learning_rate=5e-4, 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True, # Critical for T4 GPU memory [cite: 48]
    push_to_hub=False
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=lambda x: compute_metrics(x, tokenizer)
)

# Start Fine-Tuning
print("Fine-tuning started...")
trainer.train()

# Save the final LoRA Adapter
model.save_pretrained("./final_am_en_translator_lora")
print("Model saved as final_am_en_translator_lora")