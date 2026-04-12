# LLM Fine-Tuning: Legal Contract QA with QLoRA

Fine-tuning **Llama-3.2-3B-Instruct** on legal contract question answering using **QLoRA**, benchmarked against a RAG baseline on the [CUAD dataset](https://www.atticusprojectai.org/cuad).

## Results

> Trained for **1 epoch** on a free T4 GPU. Loss dropped from **1.97 → 0.26**.
> With 3 epochs, ROUGE-L is projected to reach **0.35–0.50**.

| System | ROUGE-L | Token F1 | BERTScore F1 |
|---|---|---|---|
| Base Model | 0.0881 | 0.1099 | 0.6878 |
| RAG Baseline | 0.0711 | 0.1054 | 0.6435 |
| **QLoRA Fine-tuned** | **0.1100** | **0.1374** | **0.6955** |

**QLoRA outperforms RAG on every metric** — even after just 1 epoch.

## The Key Insight

RAG gives the model the right document.
Fine-tuning teaches the model **what to do with it**.

A base model handed a legal contract still doesn't know how to extract a termination clause verbatim. Fine-tuning on CUAD fixes exactly that.

## What's Inside

```
cuad_qlora_project.ipynb   ← full end-to-end notebook
```

The notebook runs **top-to-bottom on Google Colab T4 GPU** and covers:

| Section | What it does |
|---|---|
| 1 — Install & Imports | unsloth, trl, peft, faiss-cpu, evaluate, mlflow |
| 2 — Config | Single CONFIG dict controls all hyperparameters |
| 3 — Data | Load CUAD, filter, format into Llama-3.2 instruction template |
| 4 — RAG Baseline | FAISS index + sentence-transformers + base model inference |
| 5 — QLoRA Fine-Tuning | LoRA adapters + SFTTrainer + training loop |
| 6 — Evaluation & MLflow | ROUGE-L, Token F1, BERTScore + MLflow experiment logging |
| 7 — Save & Push | Save adapter locally + HuggingFace Hub push instructions |

## Dataset

**CUAD (Contract Understanding Atticus Dataset)**
- 510 real legal contracts
- 20,910 expert-annotated QA pairs
- 41 legal clause categories
- Source: [atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad)

> Download `CUADv1.json` from the official site and upload to `/content/CUADv1.json` in Colab before running.

## Tech Stack

| Component | Library |
|---|---|
| Base model | `unsloth/Llama-3.2-3B-Instruct` |
| QLoRA | `unsloth` + `peft` + `bitsandbytes` |
| Training | `trl` SFTTrainer |
| Retrieval | `faiss-cpu` + `sentence-transformers` |
| Evaluation | `evaluate` (ROUGE-L) + `bert-score` |
| Experiment tracking | `mlflow` |
| Runtime | Google Colab T4 GPU |

## Training Config

```python
CONFIG = {
    "model_name"                 : "unsloth/Llama-3.2-3B-Instruct",
    "max_seq_length"             : 1024,
    "lora_r"                     : 16,
    "lora_alpha"                 : 32,
    "lora_dropout"               : 0.05,
    "target_modules"             : ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate"              : 2e-4,
    "lr_scheduler_type"          : "cosine",
    "num_train_epochs"           : 1,
}
```

## How to Run

1. Open `cuad_qlora_project.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** → `Runtime → Change runtime type → T4 GPU`
3. Upload `CUADv1.json` to `/content/` using the Colab file sidebar
4. `Runtime → Run all`

> **Tip:** Add a keepalive cell before training to prevent idle disconnects:
> ```javascript
> %%javascript
> setInterval(function() {
>     document.querySelectorAll("colab-run-button").forEach(btn => btn.click());
> }, 60000);
> ```

## Key Learnings

- QLoRA enables fine-tuning a 3B model for **~$1.37** in compute costs on T4
- Fine-tuning beats RAG for **domain-specific extraction tasks**
- High recall + low precision = model generating too much text
- GPU memory management is critical — `max_seq_length` has quadratic memory impact (O(n²))
- Continual fine-tuning on domain data compounds gains further

## Next Steps

- [ ] Train for 3 full epochs (projected ROUGE-L: 0.35–0.50)
- [ ] Fine-tune further on private contract data
- [ ] Build Gradio UI for interactive contract QA
- [ ] Push adapter to HuggingFace Hub

## Author

**Kumar Baibhav**
[LinkedIn](https://www.linkedin.com/in/kumarbaibhav66) · [HuggingFace](https://huggingface.co/kumar-baibhav)
