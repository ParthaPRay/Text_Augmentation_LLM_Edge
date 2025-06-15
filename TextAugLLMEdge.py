# Text Augmentation in Localized Edge

# Partha Pratim Ray, 15/06/2025, parthapratimray1986@gmail.com

# uvicorn TextAugLLMEdge:app --host 0.0.0.0 --port 8000

# curl -X POST http://localhost:8000/run-batch



from fastapi import FastAPI
from fastapi.responses import JSONResponse
import requests
import time
import csv
from datetime import datetime
from difflib import SequenceMatcher
import os

# BLEU is optional: set to True to enable
USE_BLEU = True 
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("Install nltk for BLEU: pip install nltk")
    USE_BLEU = False 

# Cosine Similarity for word-counts (TF-IDF is overkill for short text, use count-based)
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Install scikit-learn: pip install scikit-learn")
    CountVectorizer = None
    cosine_similarity = None

app = FastAPI()

# ---- CONFIG ----

# <<<< EDIT THIS LINE ONLY TO CHANGE MODEL >>>>
MODEL = "granite3.1-moe:1b"  # Change here as desired
# <<<< ------------------------------------ >>>>

AUGMENT_TYPES = {
    "paraphrase": "Paraphrase the following sentence: {text}",
    "synonym": "Replace at least two words in the following sentence with synonyms: {text}",
    "explain_simple": "Explain the following sentence to a 10-year-old: {text}",
    "summarize": "Summarize the following text in one sentence: {text}",
    "expand": "Expand the following sentence with more detail: {text}",
    "question_gen": "Generate a question from this sentence: {text}",
    "noise": "Add two random spelling mistakes to this sentence: {text}",
    "shuffle": "Reorder the words of this sentence randomly but keep it grammatical: {text}",
    "entity_replace": "Replace any names or places in this sentence with different names or places: {text}",
    "negation": "Rewrite this sentence with the opposite meaning: {text}",
    "identity": "Return the sentence unchanged: {text}" # For calibration/debug
}


TEST_PROMPTS = [
    # Agriculture/Economics, factual and complex
    "Although agriculture remains the backbone of our nation's economy, providing employment and sustenance to millions, it faces significant challenges due to climate change, unpredictable monsoon patterns, and evolving market demands that threaten farmers' livelihoods and food security.",
    
    # Technology/Education, progress-oriented
    "The rapid advancement of technology has not only transformed the way people communicate and access information, but has also made digital literacy and reliable internet connectivity essential prerequisites for personal development and economic opportunity in the 21st century.",
    
    # Social Good/Decision, narrative
    "Despite receiving a prestigious scholarship to study abroad, Meera chose to remain in her home village to establish a learning center for underprivileged children, believing that true societal progress begins with empowering the grassroots.",
    
    # Disaster Relief/Collaboration, event-driven
    "In the aftermath of the devastating earthquake, government agencies, humanitarian organizations, and local volunteers worked together tirelessly to provide shelter, medical care, and essential supplies to thousands of displaced families, demonstrating the power of collective action during times of crisis.",
    
    # Environment/Sustainability, policy-oriented
    "The increasing adoption of renewable energy sources, such as solar and wind power, is critical not only for reducing dependence on fossil fuels but also for mitigating the adverse impacts of climate change and ensuring a more sustainable future for coming generations.",
    
    # Medicine/Science, technical style
    "Recent advances in medical research, particularly in genomics and personalized medicine, have paved the way for innovative therapies that target the underlying causes of disease, yet they also raise complex ethical and regulatory challenges that must be addressed.",
    
    # Cultural/Societal, abstract reasoning
    "Cultural diversity enriches societies by fostering creativity and innovation, but it also demands mutual respect and open dialogue to address potential misunderstandings and conflicts arising from differing values, traditions, and worldviews.",
    
    # Law/Governance, complex cause-effect
    "Comprehensive legal reforms are necessary to ensure that technological developments in artificial intelligence are deployed responsibly, balancing the benefits of automation with the protection of human rights, privacy, and societal well-being."
]



OLLAMA_URL = "http://localhost:11434/api/generate"
CSV_DIR = "."


# ---- METRICS ----

def levenshtein_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def jaccard_similarity(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0

def length_ratio(a, b):
    return len(b) / len(a) if len(a) > 0 else 0

def bleu_score(a, b):
    try:
        # Use simple whitespace split for tokenization for robust, dependency-free BLEU
        ref = a.split()
        hyp = b.split()
        if not ref or not hyp:
            return 0.0
        smoothie = SmoothingFunction().method4
        return sentence_bleu([ref], hyp, smoothing_function=smoothie)
    except Exception:
        return 0.0


def cosine_sim_word_counts(a, b):
    if not a.strip() or not b.strip() or not CountVectorizer:
        return 0.0
    try:
        vectorizer = CountVectorizer().fit([a, b])
        vecs = vectorizer.transform([a, b])
        return float(cosine_similarity(vecs[0], vecs[1])[0][0])
    except Exception:
        return 0.0

def word_error_rate(a, b):
    ref = a.split()
    hyp = b.split()
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(ref)][len(hyp)] / len(ref) if len(ref) > 0 else 0.0

def char_diversity(a, b):
    chars_a = set(a)
    chars_b = set(b)
    unique_chars_in_b = chars_b - chars_a
    return len(unique_chars_in_b) / len(chars_b) if len(chars_b) > 0 else 0.0

def type_token_ratio(text):
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def ngram_overlap(a, b, n=2):
    def ngrams(text, n):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    ngrams_a = ngrams(a, n)
    ngrams_b = ngrams(b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

def get_csv_path():
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"ollama_aug_results_{MODEL.replace(':','_')}_{now}.csv"


@app.post("/run-batch")
def run_batch():
    print("\n[DEBUG] Starting Ollama Augmentation Batch")
    print(f"[DEBUG] Using Model: {MODEL}")
    print(f"[DEBUG] {len(TEST_PROMPTS)} prompts x {len(AUGMENT_TYPES)} augmentation types = {len(TEST_PROMPTS)*len(AUGMENT_TYPES)} runs\n")
    csv_path = get_csv_path()
    header = [
        "timestamp", "prompt", "augmentation_type", "model", "augmented_text",
        "total_duration_ns", "load_duration_ns", "prompt_eval_count", "prompt_eval_duration_ns",
        "eval_count", "eval_duration_ns", "tokens_per_second",
        "levenshtein_similarity", "jaccard_similarity", "length_ratio",
        "bleu", "cosine_similarity", "wer", "char_diversity",
        "type_token_ratio", "bigram_overlap"
    ]
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[DEBUG] CSV header written to {csv_path}")

    all_results = []
    count = 0
    for aug_type, prompt_template in AUGMENT_TYPES.items():
        print(f"[DEBUG] Augmentation Type: {aug_type}")
        for prompt in TEST_PROMPTS:
            count += 1
            time_now = datetime.now().isoformat()
            prompt_text = prompt_template.format(text=prompt)
            print(f"\n[DEBUG] #{count} Prompt: '{prompt}'")
            print(f"[DEBUG]    Augmentation Prompt: '{prompt_text}'")
            payload = {
                "model": MODEL,
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 60,
                    "top_k": 20,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2
                }
            }
            try:
                print("[DEBUG]    Sending request to Ollama...")
                resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
                data = resp.json()
                output = data.get("response", "").strip()
                print(f"[DEBUG]    Received response: '{output}'")
                metrics = {
                    k: data.get(k, "") for k in [
                        "total_duration", "load_duration", "prompt_eval_count",
                        "prompt_eval_duration", "eval_count", "eval_duration"
                    ]
                }
                metrics["tokens_per_second"] = (
                    float(metrics["eval_count"]) / float(metrics["eval_duration"]) * 1e9
                    if metrics["eval_count"] and metrics["eval_duration"] and float(metrics["eval_duration"]) > 0
                    else 0
                )
                print("[DEBUG]    Calculating metrics...")
                lev = levenshtein_ratio(prompt, output)
                jac = jaccard_similarity(prompt, output)
                lenr = length_ratio(prompt, output)
                bleu = bleu_score(prompt, output) if USE_BLEU else 0.0
                cosine_sim = cosine_sim_word_counts(prompt, output)
                wer = word_error_rate(prompt, output)
                chardiv = char_diversity(prompt, output)
                ttr = type_token_ratio(output)
                bigram_ol = ngram_overlap(prompt, output, 2)
                print(f"[DEBUG]    Metrics: Levenshtein={lev:.3f}, Jaccard={jac:.3f}, Length Ratio={lenr:.3f}, BLEU={bleu:.3f}, Cosine={cosine_sim:.3f}, WER={wer:.3f}, CharDiv={chardiv:.3f}, TTR={ttr:.3f}, BigramOver={bigram_ol:.3f}")
                row = [
                    time_now, prompt, aug_type, MODEL, output,
                    metrics["total_duration"], metrics["load_duration"], metrics["prompt_eval_count"],
                    metrics["prompt_eval_duration"], metrics["eval_count"], metrics["eval_duration"], metrics["tokens_per_second"],
                    lev, jac, lenr, bleu, cosine_sim, wer, chardiv, ttr, bigram_ol
                ]
                # Save row
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"[DEBUG]    Row written to CSV.")
                all_results.append({
                    "timestamp": time_now,
                    "prompt": prompt,
                    "augmentation_type": aug_type,
                    "model": MODEL,
                    "augmented_text": output,
                    "ollama_metrics": metrics,
                    "levenshtein_similarity": lev,
                    "jaccard_similarity": jac,
                    "length_ratio": lenr,
                    "bleu": bleu,
                    "cosine_similarity": cosine_sim,
                    "wer": wer,
                    "char_diversity": chardiv,
                    "type_token_ratio": ttr,
                    "bigram_overlap": bigram_ol
                })
            except Exception as e:
                err = f"ERROR: {e}"
                print(f"[DEBUG]    Exception: {err}")
                all_results.append({
                    "timestamp": time_now,
                    "prompt": prompt,
                    "augmentation_type": aug_type,
                    "model": MODEL,
                    "augmented_text": err,
                    "ollama_metrics": {},
                    "levenshtein_similarity": 0.0,
                    "jaccard_similarity": 0.0,
                    "length_ratio": 0.0,
                    "bleu": 0.0,
                    "cosine_similarity": 0.0,
                    "wer": 0.0,
                    "char_diversity": 0.0,
                    "type_token_ratio": 0.0,
                    "bigram_overlap": 0.0
                })

    print(f"\n[DEBUG] Batch Complete! {count} generations performed.")
    print(f"[DEBUG] Results saved to: {csv_path}\n")

    return JSONResponse(
        {"status": "completed", "csv_file": csv_path, "total": len(all_results), "sample": all_results[:2]}
    )

@app.get("/")
def read_root():
    return {"message": "Ollama Text Augmentation API. POST to /run-batch to trigger the experiment."}
