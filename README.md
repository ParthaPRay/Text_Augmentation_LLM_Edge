# TextAugLLMEdge: Localized Text Augmentation on Edge Devices with Quantized LLMs

_Author: Partha Pratim Ray (ppray@cus.ac.in)_

## Overview

**TextAugLLMEdge** is a Python FastAPI application that performs advanced text augmentation using small, quantized large language models (LLMs) served locally on a Raspberry Pi 4B (or similar resource-constrained edge device) via [Ollama](https://ollama.com/).  
The framework supports LLMs such as `gemma3:1b`, `granite3.1-moe:1b`, `qwen2:0.5b`, and `smollm2:360m`, and generates a comprehensive CSV with augmentation outputs and diverse linguistic/throughput metrics.

---

## Requirements

Create a file called `requirements.txt`:

```txt
fastapi
uvicorn
requests
nltk
scikit-learn
````

Then install dependencies:

```bash
pip install -r requirements.txt
```

For BLEU metrics, you must also download NLTK's 'punkt' tokenizer:

```python
import nltk
nltk.download('punkt')
```

---

## Ollama Setup

Ensure [Ollama](https://ollama.com/) is **installed, models are pulled**, and the Ollama server is running, e.g.:

```bash
ollama serve
ollama pull gemma3:1b
ollama pull granite3.1-moe:1b
ollama pull qwen2:0.5b
ollama pull smollm2:360m
```

---

## How to Run

### 1. **Start the API Server**

From the folder containing `TextAugLLMEdge.py`, run:

```bash
uvicorn TextAugLLMEdge:app --host 0.0.0.0 --port 8000
```

### 2. **Trigger the Augmentation Batch**

From another terminal (local or remote), trigger a batch run:

```bash
curl -X POST http://localhost:8000/run-batch
```

* The current model is set via the `MODEL` variable in the script (`TextAugLLMEdge.py`). Change this to switch LLMs.

---

## Features

* **Batch Augmentation:** Multiple linguistic augmentation types (paraphrase, synonym, explain simple, summarize, etc.)
* **Flexible LLM Backend:** Easily switch between locally hosted quantized models (just edit the `MODEL` variable).
* **Diverse Prompts:** Covers domains like Agriculture, Technology, Medicine, Law, etc.
* **Comprehensive Metrics:** Output CSV contains:

  * Augmentation metadata (type, prompt, model)
  * Output text
  * Ollama timing metrics
  * Semantic/lexical similarity (Levenshtein, Jaccard, BLEU, Cosine)
  * Diversity (type-token ratio, char diversity, bigram overlap)
  * Throughput (tokens/sec), error rates, and more

---

## Output

* Each run produces a timestamped CSV file, e.g. `ollama_aug_results_granite3.1-moe_20250615_154200.csv`.
* Each row = (prompt, augmentation type, LLM, generated text, metrics).

---

## Code Quickstart

### BLEU Setup (first run):

```python
import nltk
nltk.download('punkt')
```

### Example Server Command

```bash
uvicorn TextAugLLMEdge:app --host 0.0.0.0 --port 8000
```

### Example CURL Trigger

```bash
curl -X POST http://localhost:8000/run-batch
```

---

## Customization

* To **change the LLM model**, set the `MODEL` variable in the script.
* To add/remove **augmentation types**, edit the `AUGMENT_TYPES` dict.
* To change **prompts**, edit the `TEST_PROMPTS` list.
* All output CSVs are saved in the current directory.

---

## Citation

If you use or extend this code for research, please cite:

> Partha Pratim Ray, Mohan Pratap Pradhan, "TextAugLLMEdge: A Text Augmentation Framework Using Localized Large Language Models for Resource-Constrained Edge," (2025).
> Department of Computer Applications, Sikkim University, India.

---

## License

MIT License (or specify your actual license here).

---

## Contact

For questions or collaborations, contact [ppray@cus.ac.in](mailto:ppray@cus.ac.in).

---

