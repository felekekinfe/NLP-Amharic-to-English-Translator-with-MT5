```markdown
# Amharic-English Machine Translation with mT5 & LoRA

This repository contains the complete pipeline for fine-tuning a multilingual Text-to-Text Transfer Transformer (mT5) to translate Amharic to English. The project leverages **Low-Rank Adaptation (LoRA)** to achieve high-performance results on consumer-grade hardware (Google Colab T4 GPU).

---

## 🚀 Project Overview

Amharic is a "low-resource" language with significant morphological complexity. This project addresses these challenges through a specialized preprocessing pipeline and parameter-efficient fine-tuning (PEFT).

### Key Features
* **Base Model**: `google/mt5-small` (Encoder-Decoder architecture).
* **Efficiency**: **LoRA** (Rank 16, Alpha 32) updates <1% of total parameters.
* **Preprocessing**: Phonetic-aware Unicode Normalization for the Ge'ez script.
* **Evaluation**: Multi-angle assessment using **BLEU** and **chrF** scores.

---

## 📂 Project Structure

* `preprocessing.py`: Contains the `clean_and_normalize_dataset` function for Unicode mapping and data cleaning.
* `main.py`: The fine-tuning script utilizing Hugging Face `Trainer` and `peft` for LoRA training.
* `dataset/`: Directory containing dataset files.
* Various CSV files: `cleaned_dataset.csv`, `cleaned_master.csv`, `converted_test.csv`, `converted_train_1.csv`, `converted_train.csv`, `preprocessed_dataset.csv` - processed datasets at different stages.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training, though CPU training is possible)

### Dependencies
```bash
pip install pandas torch numpy datasets transformers peft evaluate sacrebleu chrf sentencepiece
```

---

## 📖 Methodology

### 1. Unicode Normalization
Amharic features redundant characters (e.g., ሐ, ኀ, ሀ) that represent the same sound. Our preprocessing script maps these to a canonical form, which:
* Reduces vocabulary sparsity.
* Prevents the model from treating identical phonetic concepts as different words.
* Cleans noisy symbols and standardizes whitespace.

### 2. Model Architecture (mT5)
We utilize an **Encoder-Decoder** structure. Unlike encoder-only models (like XLM-R), mT5 is purpose-built for generative tasks. The encoder extracts semantic features from Amharic, while the decoder auto-regressively generates English text.

### 3. LoRA Fine-Tuning
To enable training on a 16GB T4 GPU, we implement **Low-Rank Adaptation**:
* **Target Modules**: Query (`q`) and Value (`v`) matrices in the attention layers.
* **Performance**: Prevents "Catastrophic Forgetting" and maintains 99% of full fine-tuning performance while saving massive VRAM.

---

## 📊 Evaluation
We employ a dual-metric evaluation strategy:
* **BLEU Score**: For word-level precision.
* **chrF Score**: A character-level n-gram F-score. This is critical for Amharic as it captures partial matches, making it more robust against complex morphological variations.

---

## 📊 Dataset

The project includes preprocessed datasets in CSV format with the following structure:
- `input`: Amharic text to be translated
- `output`: Corresponding English translation

Example datasets provided:
- `converted_train.csv`, `converted_train_1.csv`: Raw training data
- `converted_test.csv`: Test data
- `cleaned_dataset.csv`: Preprocessed and cleaned dataset ready for training
- `preprocessed_dataset.csv`: Final processed dataset with prefix added

### Data Format
Input CSV files should have two columns: `input` (Amharic) and `output` (English). The preprocessing script will automatically add the translation prefix "translate Amharic to English: " to input texts.

---

## 🏁 How to Use

### Step 1: Preprocess Data
Prepare your parallel corpora in CSV format (columns: `input`, `output`) and run:
```python
from preprocessing import clean_and_normalize_dataset
clean_and_normalize_dataset('converted_train.csv', 'converted_train_1.csv', 'cleaned_dataset.csv')
```

### Step 2: Training
Run the training script to inject LoRA adapters and begin fine-tuning:
```bash
python main.py
```

### Step 3: Inference
Load the base mT5 model and your saved LoRA adapters for translation:
```python
from peft import PeftModel
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model_id = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_id)
model = MT5ForConditionalGeneration.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, "./final_am_en_translator_lora")

# Example translation
input_text = "እንዴት ነህ?"
inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
outputs = model.generate(**inputs, max_length=128)
translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated)
```

---

## ⚙️ Training Configuration

The training is configured for efficient fine-tuning on limited hardware:
- **Batch Size**: 8 (train/eval)
- **Learning Rate**: 5e-4
- **Epochs**: 3
- **Max Length**: 128 tokens
- **LoRA Rank**: 16, Alpha: 32
- **GPU**: T4 (16GB VRAM) with FP16 precision

---

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Areas for Improvement
- Experiment with different LoRA configurations
- Add more evaluation metrics
- Support for other Ethiopian languages
- Integration with translation APIs

