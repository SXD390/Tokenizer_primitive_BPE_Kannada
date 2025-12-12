# 🌿 Kannada BPE Tokenizer

A custom **Byte Pair Encoding (BPE)** tokenizer trained from scratch on a rich corpus of Kannada literature.
The tokenizer learns meaningful subword units directly from text and provides efficient segmentation, high compression, and expressive tokenization for Kannada NLP tasks.

---

## ✨ Features

* 🔤 **Fully custom BPE implementation** (no external tokenizer libraries)
* 📚 Trained on **Kannada novels and long-form prose**
* 🔍 Learns subword patterns directly from language statistics
* ⚡ **High compression ratio** → efficient tokenization
* 🎛️ Easy-to-use **encode** and **decode** API
* 🌐 Interactive demo on Hugging Face Spaces
* 📦 Lightweight, CPU-friendly, pure-Python implementation

---

## 🎥 Demo



![Demo](https://github.com/SXD390/Tokenizer_primitive_BPE_Kannada/blob/main/DATA/util/KN_BPE_Tokenizer_DEMO.gif)


---

## 📊 Training Summary

| Metric                    | Value                                      |
| ------------------------- | ------------------------------------------ |
| **Corpus size**           | ~140k characters (combined Kannada novels) |
| **Final vocabulary size** | **9002 tokens**                            |
| **Compression ratio**     | **3.7543** (chars / tokens)                |
| **Tokenizer type**        | Character-level BPE                        |
| **Training hardware**     | CPU-only                                   |

### 📘 Compression Ratio Explained

Compression ratio tells how efficiently text is tokenized:

The compression ratio can be defined by the formula: $\text{compression ratio} = \frac{\text{total characters}}{\text{total tokens}}$.



A ratio of **3.75** means:
**Each token represents 3.75 original characters on average** → excellent efficiency for Kannada.

---

## 🚀 Try the Tokenizer

### ▶️ Interactive Web Demo

Use the tokenizer directly in your browser:

👉 **Hugging Face Space:**

```
https://huggingface.co/spaces/SXD390/BPE_KN_Tokenizer
```

### Example (Encoding)

**Input:**

```
ನಮಸ್ಕಾರ. ನೀವು ಹೇಗಿದ್ದೀರಿ?
```

**Output tokens:**
`[2123, 981, 7740, ...]`

**Compression ratio:**
`3.82`

### Example (Decoding)

```
[2123, 981, 7740, ...] → "ನಮಸ್ಕಾರ. ನೀವು ಹೇಗಿದ್ದೀರಿ?"
```

*(Note: decoding is approximate because BPE merges tokens irreversibly.)*

---

## 🧠 How It Works

### 1️⃣ Initial Character Vocabulary

The tokenizer begins with all unique Kannada Unicode characters plus an end-of-word marker.

### 2️⃣ Pair Frequency Analysis

It scans the entire corpus to find the **most frequent adjacent character pairs**.

### 3️⃣ Merge Operations

The most common pairs are merged into new tokens.
This process repeats until the target vocabulary size is reached.

### 4️⃣ Tokenization

When encoding:

* Words are split into characters
* The learned merger rules are applied greedily
* Output tokens represent meaningful subword units

---

## 📁 Project Structure

```
project/
├── train_kannada_bpe.py      # BPE implementation + training script
├── train_tokenizer.ipynb     # Notebook demonstration & reproducibility
├── artifacts/
│   ├── vocab.json            # Learned vocabulary
│   └── merges.json           # Learned BPE merge rules
├── hf_space/
│   ├── app.py                # Gradio app
│   ├── requirements.txt
│   ├── vocab.json
│   └── merges.json
└── txt_out/                  # Processed text files (if included)
```

---

## 🛠 Usage

### Install

```bash
pip install gradio
```

### Load the tokenizer

```python
from train_kannada_bpe import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.load("vocab.json", "merges.json")
```

### Encode

```python
ids = tokenizer.encode("ನಮಸ್ಕಾರ")
print(ids)
```

### Decode

```python
text = tokenizer.decode(ids)
print(text)
```

---

## 📚 Data Source

The tokenizer was trained on a manually assembled collection of publicly available Kannada novels in PDF → text format.
Only the processed text is used; PDFs are not required.

---

---
## 🪪 License

MIT License
