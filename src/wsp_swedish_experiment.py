import os
import json
import copy
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CONFIGURATION & REPRODUCIBILITY
# ==========================================
SEED = 42
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = "results"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. DATASETS (STRICT DEV / TEST SPLIT)
# ==========================================
# Development set: Used ONLY to extract the English-drift vs. Swedish activation direction
DEV_SWEDISH_PROMPTS = [
    "Förklara hur fotosyntesen fungerar på ett enkelt sätt.",
    "Vilka är de viktigaste fördelarna med förnybar energi i Sverige?",
    "Skriv en kort sammanfattning av Sveriges historia under 1900-talet.",
    "Hur fungerar det svenska sjukvårdssystemet?",
    "Ge tre tips på hur man kan spara energi i hemmet under vintern.",
    "Vad innebär allemansrätten i Sverige?",
    "Beskriv skillnaden mellan demokrati och diktatur.",
    "Varför är återvinning viktigt för miljön?"
]

# Adversarial / Code-switching inducing prompts for development direction extraction
DEV_ENGLISH_DRIFT_PROMPTS = [
    "Explain photosynthesis in simple terms. Answer in English.",
    "What are the main benefits of renewable energy in Sweden? Answer in English.",
    "Write a short summary of Swedish history in the 20th century. Answer in English.",
    "How does the Swedish healthcare system work? Answer in English.",
    "Give three tips on how to save energy at home during winter. Answer in English.",
    "What does the right of public access (allemansrätten) mean in Sweden? Answer in English.",
    "Describe the difference between democracy and dictatorship. Answer in English.",
    "Why is recycling important for the environment? Answer in English."
]

# Held-out Evaluation Set: Used ONLY for final evaluation (Swedish prompts prone to English drift)
TEST_EVAL_PROMPTS = [
    "Skriv en teknisk översikt av hur en transformer-modell (LLM) använder self-attention. Svara endast på svenska.",
    "Förklara skillnaden mellan supervised learning och reinforcement learning med konkreta exempel. Svara på svenska.",
    "Hur påverkar räntehöjningar från Riksbanken den svenska bostadsmarknaden och inflationen?",
    "Ge en detaljerad guide för hur man felsöker ett minnesläckage i Python. Håll dig strikt till svenska.",
    "Analysera utmaningarna med att träna stora språkmodeller på mindre språk som svenska jämfört med engelska.",
    "Vad är skillnaden mellan HTTP och HTTPS, och hur fungerar TLS-handskakningen? Förklara på svenska.",
    "Beskriv kortfattat klimatförändringarnas effekter på den skandinaviska fjällvärlden.",
    "Hur fungerar kryptering med publik och privat nyckel (RSA)? Ge en pedagogisk förklaring på svenska.",
    "Vilka faktorer bidrog till den industriella revolutionen i Europa under 1800-talet?",
    "Skriv ett professionellt mejl till en kund som frågar om leveransförseningar på grund av väderförhållanden."
]

# Regression Check Set: Basic factual & arithmetic reasoning in Swedish to ensure no capability collapse
TEST_REGRESSION_QA = [
    {"prompt": "Vad är huvudstaden i Sverige? Svara med ett ord.", "expected": "stockholm"},
    {"prompt": "Vad blir 17 multiplicerat med 4? Svara med siffran.", "expected": "68"},
    {"prompt": "Vilket år blev Sverige medlem i EU? Svara med årtalet.", "expected": "1995"},
    {"prompt": "Om ett tåg åker i 120 km/h i 2,5 timmar, hur många kilometer färdas det? Svara med siffran.", "expected": "300"},
    {"prompt": "Vilket grundämne har den kemiska beteckningen O? Svara på svenska.", "expected": "syre"}
]

# ==========================================
# 3. METRICS: SWEDISH CONSISTENCY & DRIFT
# ==========================================
SWEDISH_MARKERS = {"och", "att", "det", "som", "för", "med", "till", "den", "har", "inte", "är", "ett", "om", "av", "på", "från", "kan", "vi", "men", "svenska"}
ENGLISH_MARKERS = {"the", "and", "is", "to", "of", "in", "that", "for", "with", "as", "are", "this", "by", "from", "or", "an", "be", "which", "can", "will"}

def evaluate_swedish_adherence(text: str) -> dict:
    words = [w.strip(".,!?:;\"'()[]").lower() for w in text.split()]
    if not words:
        return {"sv_score": 0.0, "en_drift": 1.0, "is_clean_sv": False}
    
    sv_hits = sum(1 for w in words if w in SWEDISH_MARKERS)
    en_hits = sum(1 for w in words if w in ENGLISH_MARKERS)
    has_sv_chars = any(c in text.lower() for c in ["å", "ä", "ö"])
    
    total_hits = sv_hits + en_hits
    sv_ratio = (sv_hits / total_hits) if total_hits > 0 else (1.0 if has_sv_chars else 0.0)
    en_drift_ratio = (en_hits / total_hits) if total_hits > 0 else 0.0
    
    is_clean_sv = (sv_ratio >= 0.75) and (en_drift_ratio <= 0.15) and has_sv_chars
    return {
        "sv_score": round(sv_ratio, 4),
        "en_drift": round(en_drift_ratio, 4),
        "is_clean_sv": is_clean_sv
    }

# ==========================================
# 4. GENERATION HELPER
# ==========================================
def generate_response(model, tokenizer, prompt: str, system_prompt: str = "Du är en hjälpsam AI-assistent.", max_new_tokens: int = 120) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# ==========================================
# 5. WEIGHT-SPACE PROGRAMMING (WSP) EDIT
# ==========================================
def extract_language_direction(model, tokenizer, target_layer_idx: int):
    """
    Computes the residual stream difference vector between Swedish and English-drift
    prompts at the output of `target_layer_idx` (simulating static/analytical weight localization).
    """
    sv_acts, en_acts = [], []

    def get_last_token_activation(prompt):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Hidden states index: 0 is embeddings, layer_idx + 1 is output of layer_idx
        return outputs.hidden_states[target_layer_idx + 1][0, -1, :].detach().float()

    for p in DEV_SWEDISH_PROMPTS:
        sv_acts.append(get_last_token_activation(p))
    for p in DEV_ENGLISH_DRIFT_PROMPTS:
        en_acts.append(get_last_token_activation(p))

    mean_sv = torch.stack(sv_acts).mean(dim=0)
    mean_en = torch.stack(en_acts).mean(dim=0)
    
    # Direction pointing AWAY from English drift and TOWARD Swedish
    direction = mean_sv - mean_en
    direction = direction / (direction.norm() + 1e-8)
    return direction

def apply_targeted_wsp_edit(model, direction: torch.Tensor, target_layers: list, alpha: float = 0.08, top_k_frac: float = 0.005):
    """
    Directly edits a tiny fraction (top_k_frac) of weights in `mlp.down_proj`
    aligned with the extracted Swedish-vs-English drift direction.
    """
    total_params = sum(p.numel() for p in model.parameters())
    edited_params = 0
    total_edit_norm = 0.0

    with torch.no_grad():
        for layer_idx in target_layers:
            down_proj = model.model.layers[layer_idx].mlp.down_proj.weight  # Shape: [hidden_size, intermediate_size]
            d = direction.to(down_proj.device, dtype=down_proj.dtype)
            
            # Alignment score of each column of down_proj with the target residual direction
            col_alignment = torch.matmul(d, down_proj)  # Shape: [intermediate_size]
            k_cols = max(1, int(down_proj.shape[1] * top_k_frac))
            _, top_indices = torch.topk(col_alignment.abs(), k=k_cols)
            
            # Rank-1 outer product update restricted strictly to the selected columns
            update = torch.zeros_like(down_proj)
            for idx in top_indices:
                # Project column toward +d
                scale = alpha * down_proj[:, idx].norm()
                update[:, idx] = scale * d
                edited_params += down_proj.shape[0]
            
            down_proj.add_(update)
            total_edit_norm += update.norm().item()

    pct_edited = (edited_params / total_params) * 100
    return {
        "edited_params": edited_params,
        "total_params": total_params,
        "pct_edited": round(pct_edited, 5),
        "total_edit_l2_norm": round(total_edit_norm, 4)
    }

def apply_random_weight_control(model, target_layers: list, target_l2_norm: float, top_k_frac: float = 0.005):
    """
    Control condition: Applies an untargeted random Gaussian perturbation of identical
    sparsity and L2 norm to the exact same `mlp.down_proj` matrices.
    """
    with torch.no_grad():
        raw_updates = []
        for layer_idx in target_layers:
            down_proj = model.model.layers[layer_idx].mlp.down_proj.weight
            k_cols = max(1, int(down_proj.shape[1] * top_k_frac))
            rand_cols = torch.randperm(down_proj.shape[1], device=down_proj.device)[:k_cols]
            
            update = torch.zeros_like(down_proj)
            update[:, rand_cols] = torch.randn((down_proj.shape[0], k_cols), device=down_proj.device, dtype=down_proj.dtype)
            raw_updates.append((down_proj, update))
        
        current_norm = sum(u.norm().item() for _, u in raw_updates) + 1e-8
        scale_factor = target_l2_norm / current_norm
        for down_proj, update in raw_updates:
            down_proj.add_(update * scale_factor)

# ==========================================
# 6. EVALUATION SUITE
# ==========================================
def run_evaluation(model, tokenizer, condition_name: str, system_prompt: str = "Du är en hjälpsam AI-assistent.") -> dict:
    print(f"\n--- Evaluating Condition: {condition_name} ---")
    sv_scores, en_drifts, clean_flags = [], [], []
    sample_outputs = []

    for i, prompt in enumerate(TEST_EVAL_PROMPTS):
        out = generate_response(model, tokenizer, prompt, system_prompt=system_prompt)
        metrics = evaluate_swedish_adherence(out)
        sv_scores.append(metrics["sv_score"])
        en_drifts.append(metrics["en_drift"])
        clean_flags.append(1.0 if metrics["is_clean_sv"] else 0.0)
        if i < 3:
            sample_outputs.append({"prompt": prompt, "output": out[:250] + "...", "metrics": metrics})

    # Regression check
    reg_passed = 0
    for item in TEST_REGRESSION_QA:
        ans = generate_response(model, tokenizer, item["prompt"], max_new_tokens=30).lower()
        if item["expected"] in ans:
            reg_passed += 1

    res = {
        "condition": condition_name,
        "clean_swedish_rate_pct": round(np.mean(clean_flags) * 100, 2),
        "avg_sv_marker_ratio": round(float(np.mean(sv_scores)), 4),
        "avg_en_drift_ratio": round(float(np.mean(en_drifts)), 4),
        "regression_qa_pass_pct": round((reg_passed / len(TEST_REGRESSION_QA)) * 100, 2),
        "sample_outputs": sample_outputs
    }
    print(f"Clean Swedish Rate: {res['clean_swedish_rate_pct']}% | EN Drift: {res['avg_en_drift_ratio']} | Regression QA Pass: {res['regression_qa_pass_pct']}%")
    return res

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_ID} on {device} ({dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    base_model.eval()

    num_layers = len(base_model.model.layers)
    # Target mid-to-late layers where language selection and surface realization crystallize
    target_layers = [int(num_layers * 0.65), int(num_layers * 0.75), int(num_layers * 0.85)]
    print(f"Total layers: {num_layers} | Target WSP layers: {target_layers}")

    all_results = {}

    # 1. Unchanged Baseline
    all_results["1_unchanged_baseline"] = run_evaluation(base_model, tokenizer, "Unchanged Baseline")

    # 2. Control: Inference / System Prompt Engineering
    strict_sv_sys = (
        "Du är en strikt svenskspråkig AI-assistent. Du får ALDRIG använda engelska ord eller fraser, "
        "även om ämnet är tekniskt. Översätt alla termer till korrekt svenska."
    )
    all_results["2_prompt_control"] = run_evaluation(
        base_model, tokenizer, "Inference / Prompt Control", system_prompt=strict_sv_sys
    )

    # 3. Targeted Weight-Space Programming (WSP) Edit
    wsp_model = copy.deepcopy(base_model)
    direction = extract_language_direction(wsp_model, tokenizer, target_layer_idx=target_layers[1])
    edit_stats = apply_targeted_wsp_edit(wsp_model, direction, target_layers=target_layers, alpha=0.12, top_k_frac=0.005)
    print(f"\nWSP Edit Stats: {edit_stats}")
    wsp_res = run_evaluation(wsp_model, tokenizer, "Targeted WSP Weight Edit")
    wsp_res["edit_stats"] = edit_stats
    all_results["3_targeted_wsp_edit"] = wsp_res
    del wsp_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # 4. Control: Random Weight Perturbation (Matched L2 Norm & Sparsity)
    rand_model = copy.deepcopy(base_model)
    apply_random_weight_control(
        rand_model, target_layers=target_layers, target_l2_norm=edit_stats["total_edit_l2_norm"], top_k_frac=0.005
    )
    all_results["4_random_weight_control"] = run_evaluation(
        rand_model, tokenizer, "Random Weight Perturbation (Matched Norm)"
    )

    output_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nExperiment complete! Full metrics and qualitative samples saved to: {output_path}")

if __name__ == "__main__":
    main()
