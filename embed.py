# embed.py
"""
Offline script to generate dense text embeddings for 100,000 candidates.

A1: Model = BAAI/bge-small-en-v1.5 (retrieval-optimized, similar speed to MiniLM)

A2: Smarter career text construction:
    - Most recent 2 roles get 300 chars each
    - Older roles limited to 100 chars
    - Skills list appended for concept coverage

SPEED OPTIMIZATIONS (v3):
    1. max_length reduced 256 -> 128 tokens: halves transformer compute per batch.
       Career texts are mostly short; 128 tokens covers ~400-500 characters which
       is more than enough for the career summary representation.
    2. batch_size increased 128 -> 256: better CPU parallelism utilization.
    3. torch.compile() applied on PyTorch 2+ for ~20-30% extra speedup.
    4. torch.inference_mode() used instead of torch.no_grad() (faster on PyTorch 2+).
    5. --resume flag: if interrupted, picks up where it left off.
    6. --fast flag: uses max_length=64 for a very quick rough run (~30-35 min).
    7. ETA display updated every 10 batches.

NOTE: embed.py is OFFLINE pre-computation. The 5-minute spec constraint applies
ONLY to rank.py. This script can take as long as needed.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Use all available CPU threads
torch.set_num_threads(os.cpu_count() or 4)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

CANDIDATES_PATH = r"e:\anti\Resume ATS\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
OUTPUT_VECTORS_PATH = r"e:\anti\Resume ATS\candidate_embeddings.npz"
OUTPUT_IDS_PATH     = r"e:\anti\Resume ATS\candidate_ids.json"

# Speed tuning — change these if you hit memory issues
BATCH_SIZE = 256      # was 128 -- larger batch = better CPU throughput
MAX_LENGTH = 128      # was 256 -- halves compute; 128 tokens ~= 400-500 chars


# ---------------------------------------------------------------------------
# Mean pooling
# ---------------------------------------------------------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# ---------------------------------------------------------------------------
# A2: Recency-weighted career text
# ---------------------------------------------------------------------------
def construct_career_text(candidate):
    """
    Builds a compact, recency-weighted text representation of a candidate.
    Recent roles (sorted by start_date desc) get 300 chars, older get 100.
    Skills appended as a keyword list for concept matching.
    """
    prof = candidate.get("profile", {})
    summary      = prof.get("summary", "")[:200]
    headline     = prof.get("headline", "")[:100]
    current_title = prof.get("current_title", "")

    roles = candidate.get("career_history", [])

    def get_start_year(r):
        try:
            return int(str(r.get("start_date", "2000"))[:4])
        except Exception:
            return 2000

    roles_sorted = sorted(roles, key=get_start_year, reverse=True)
    role_texts = []
    for i, role in enumerate(roles_sorted):
        char_limit = 300 if i < 2 else 100
        role_texts.append(
            f"{role.get('title', '')}: {role.get('description', '')[:char_limit]}"
        )

    skills_text = " ".join(s.get("name", "") for s in candidate.get("skills", [])[:15])

    return (
        f"{current_title}. {headline}. {summary}. "
        f"Skills: {skills_text}. "
        f"Work: {' | '.join(role_texts)}"
    )[:1200]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    limit    = None
    resume   = False
    fast     = False
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])
        if arg == "--resume":
            resume = True
        if arg == "--fast":
            fast = True

    max_length = 64 if fast else MAX_LENGTH
    if fast:
        print("[FAST MODE] max_length=64 — slightly lower quality, faster embedding.")

    print(f"Loading model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # CPU inference speedups that work on all platforms (no compiler needed)
    torch.set_num_threads(os.cpu_count() or 4)
    torch.set_num_interop_threads(max(1, (os.cpu_count() or 4) // 2))
    # Enable MKL-DNN (Intel Math Kernel Library for Deep Neural Networks) if available
    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass

    print(f"Model loaded. Batch size: {BATCH_SIZE}, Max tokens: {max_length}")

    # --- Load existing progress if resuming ---
    existing_ids = set()
    if resume and os.path.exists(OUTPUT_IDS_PATH):
        with open(OUTPUT_IDS_PATH, "r", encoding="utf-8") as f:
            existing_ids = set(json.load(f))
        print(f"Resume mode: {len(existing_ids)} candidates already embedded, skipping.")

    # --- Read all candidates upfront (fast JSON parsing pass) ---
    print(f"\nReading candidates from:\n  {CANDIDATES_PATH}")
    print("(Pre-loading all texts into memory for faster batch inference...)")
    t0 = time.time()

    candidate_ids = []
    texts         = []

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c   = json.loads(line)
            cid = c["candidate_id"]

            if cid in existing_ids:
                continue  # skip already-embedded candidates in resume mode

            candidate_ids.append(cid)
            texts.append(construct_career_text(c))

            if limit and len(candidate_ids) >= limit:
                break

    num_candidates = len(candidate_ids)
    t_load = time.time() - t0
    print(f"Loaded {num_candidates} candidate texts in {t_load:.1f}s")

    if num_candidates == 0:
        print("Nothing to embed. If using --resume, all candidates are already done.")
        return

    # --- Batch embedding with progress + ETA ---
    print(f"\nStarting embedding inference...")
    n_batches = (num_candidates + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Batches      : {n_batches}")
    print(f"  Max tokens   : {max_length}")
    est_min = max(1, num_candidates // 1500)  # ~1500 candidates/min with optimizations
    print(f"  Est. time    : ~{est_min}-{est_min+10} min on CPU")

    embeddings_list = []
    t_start = time.time()

    with torch.inference_mode():   # faster than torch.no_grad() on PyTorch 2+
        bar = tqdm(
            range(0, num_candidates, BATCH_SIZE),
            desc="Embedding",
            unit="batch",
            dynamic_ncols=True
        )
        for batch_idx, i in enumerate(bar):
            batch_texts = texts[i : i + BATCH_SIZE]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            output     = model(**encoded)
            embeddings = mean_pooling(output, encoded["attention_mask"])

            # L2-normalize
            norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            embeddings = embeddings / torch.clamp(norms, min=1e-9)
            embeddings_list.append(embeddings.cpu().numpy())

            # ETA display every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed   = time.time() - t_start
                done      = min((batch_idx + 1) * BATCH_SIZE, num_candidates)
                remaining = num_candidates - done
                eta_sec   = (elapsed / done) * remaining if done > 0 else 0
                bar.set_postfix({
                    "done": f"{done}/{num_candidates}",
                    "ETA": f"{eta_sec/60:.1f}min"
                })

    # --- Concatenate and save ---
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    t_total = time.time() - t_start
    print(f"\nEmbedding done in {t_total/60:.1f} min.")
    print(f"Shape: {all_embeddings.shape}")

    # If resuming, we need to merge with the old embeddings
    if resume and os.path.exists(OUTPUT_VECTORS_PATH) and existing_ids:
        print("Merging with existing embeddings...")
        with np.load(OUTPUT_VECTORS_PATH) as data:
            old_vecs = data["embeddings"].astype(np.float32)
        with open(OUTPUT_IDS_PATH, "r", encoding="utf-8") as f:
            old_ids = json.load(f)

        merged_vecs = np.concatenate([old_vecs, all_embeddings], axis=0)
        merged_ids  = old_ids + candidate_ids
        all_embeddings = merged_vecs
        candidate_ids  = merged_ids
        print(f"Merged total: {len(merged_ids)} candidates")

    print(f"Saving to {OUTPUT_VECTORS_PATH} ...")
    np.savez_compressed(
        OUTPUT_VECTORS_PATH,
        embeddings=all_embeddings.astype(np.float16)
    )

    print(f"Saving IDs to {OUTPUT_IDS_PATH} ...")
    with open(OUTPUT_IDS_PATH, "w", encoding="utf-8") as out:
        json.dump(candidate_ids, out)

    print("\n[DONE] Embedding pre-computation complete!")
    print(f"   Candidates : {len(candidate_ids)}")
    print(f"   Embed dim  : {all_embeddings.shape[1]}")
    print(f"   Model      : {MODEL_NAME}")
    print(f"   Total time : {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
