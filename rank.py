# rank.py
"""
Candidate Intelligence Engine - Ranking CLI (Phase 2)
Two-stage ranking architecture:
  Stage 1: Fast retrieval on all 100,000 candidates → Top 500
  Stage 2: Deep re-ranking with full scoring on Top 500 → Top 100
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from honeypot_detector import is_honeypot
from scorer import evaluate_candidate

torch.set_num_threads(os.cpu_count() or 4)

STAGE1_POOL = 500   # Number of candidates passed to Stage 2


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_docx_text(path):
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def load_job_description(base_path):
    jd_paths = [
        os.path.join(base_path, "job_description.md"),
        os.path.join(base_path, "job_description.docx"),
        os.path.join(base_path, "[PUB] India_runs_data_and_ai_challenge",
                     "India_runs_data_and_ai_challenge", "job_description.docx")
    ]
    for path in jd_paths:
        if os.path.exists(path):
            if path.endswith(".md"):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            elif path.endswith(".docx"):
                text = get_docx_text(path)
                if text:
                    return text

    print("Warning: Job description file not found. Using embedded default.")
    return (
        "Senior AI Engineer - Founding Team. Redrob AI. Series A talent intelligence platform. "
        "Must have experience with embeddings-based retrieval systems (sentence-transformers, "
        "OpenAI embeddings, BGE, E5), vector databases or hybrid search infrastructure "
        "(Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). "
        "Strong Python. Experience designing evaluation frameworks for ranking systems "
        "(NDCG, MRR, MAP). LLM fine-tuning experience (LoRA, QLoRA, PEFT). "
        "Learning-to-rank models. Pune/Noida location."
    )


def construct_career_text(candidate):
    prof = candidate.get("profile", {})
    summary = prof.get("summary", "")
    headline = prof.get("headline", "")
    current_title = prof.get("current_title", "")

    roles = candidate.get("career_history", [])
    roles_text = [
        f"{role.get('title', '')}: {role.get('description', '')}"
        for role in roles
    ]
    career_history_str = " | ".join(roles_text)
    full_text = f"{current_title} - {headline}. {summary}. Experience: {career_history_str}"
    return full_text[:1200]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates against the Senior AI Engineer JD."
    )
    parser.add_argument(
        "--candidates", type=str,
        default="./candidates.jsonl",
        help="Path to candidates JSONL dataset"
    )
    parser.add_argument(
        "--out", type=str,
        default="./submission.csv",
        help="Path to write the ranked output CSV"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------------------------
    # 1. Load Job Description
    # -------------------------------------------------------------------------
    print("Loading Job Description...")
    jd_text = load_job_description(base_dir)
    print(f"Loaded Job Description ({len(jd_text)} chars)")

    # -------------------------------------------------------------------------
    # 2. Load Sentence-Transformers Model
    # -------------------------------------------------------------------------
    print("Loading Sentence-Transformers model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # -------------------------------------------------------------------------
    # 3. Encode JD
    # -------------------------------------------------------------------------
    print("Encoding Job Description...")
    with torch.no_grad():
        encoded_jd = tokenizer(
            jd_text[:1500], padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        )
        jd_output = model(**encoded_jd)
        jd_embedding = mean_pooling(jd_output, encoded_jd["attention_mask"])[0]
        jd_embedding = jd_embedding / torch.norm(jd_embedding, p=2)
        jd_vec = jd_embedding.cpu().numpy()

    # -------------------------------------------------------------------------
    # 4. Load Pre-Computed Candidate Embeddings
    # -------------------------------------------------------------------------
    embeddings_lookup = {}
    vectors_path = os.path.join(base_dir, "candidate_embeddings.npz")
    ids_path = os.path.join(base_dir, "candidate_ids.json")

    if os.path.exists(vectors_path) and os.path.exists(ids_path):
        print(f"Loading precomputed embeddings from {vectors_path}...")
        try:
            with np.load(vectors_path) as data:
                vectors = data["embeddings"]
            with open(ids_path, "r", encoding="utf-8") as f:
                cids = json.load(f)
            if len(cids) == len(vectors):
                for cid, vec in zip(cids, vectors):
                    embeddings_lookup[cid] = vec.astype(np.float32)
                print(f"Loaded {len(embeddings_lookup)} precomputed embeddings.")
            else:
                print("Warning: ID count mismatch. Falling back to dynamic encoding.")
        except Exception as e:
            print(f"Warning: Failed to load embeddings: {e}. Falling back to dynamic encoding.")
    else:
        print("Precomputed embeddings not found. Embeddings will be computed dynamically.")

    # -------------------------------------------------------------------------
    # STAGE 1: Fast Scoring on ALL 100,000 Candidates → Top 500
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STAGE 1: Fast retrieval across all candidates...")
    print(f"{'='*60}")

    stage1_candidates = []     # stores full candidate JSON for Stage 2
    honeypot_scored = []       # honeypots kept at score 0.0

    try:
        with open(args.candidates, "rb") as f:
            num_lines = sum(1 for _ in f)
    except Exception:
        num_lines = None

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="Stage 1 Scoring"):
            if not line.strip():
                continue
            c = json.loads(line)
            cid = c["candidate_id"]

            # --- Honeypot check (always run) ---
            flagged, reasons = is_honeypot(c)
            if flagged:
                honeypot_scored.append({
                    "candidate_id": cid,
                    "score": 0.0,
                    "reasoning": (
                        f"Profile flagged during automated credentials audit. "
                        f"Reasons: {', '.join(reasons[:2])}. (Confidence: 50%)"
                    )
                })
                continue

            # --- Semantic embedding ---
            if cid in embeddings_lookup:
                c_vec = embeddings_lookup[cid]
            else:
                c_text = construct_career_text(c)
                with torch.no_grad():
                    encoded_c = tokenizer(
                        c_text, padding=True, truncation=True,
                        max_length=256, return_tensors="pt"
                    )
                    c_output = model(**encoded_c)
                    c_emb = mean_pooling(c_output, encoded_c["attention_mask"])[0]
                    c_emb = c_emb / torch.norm(c_emb, p=2)
                    c_vec = c_emb.cpu().numpy()

            semantic_score = float(np.dot(c_vec, jd_vec))

            # Stage 1 fast score (no deep text analysis)
            fast_score, _, _ = evaluate_candidate(c, semantic_score, deep=False)

            stage1_candidates.append({
                "candidate_id": cid,
                "data": c,
                "semantic_score": semantic_score,
                "stage1_score": fast_score
            })

    # Sort and select Top 500 for deep analysis
    stage1_candidates.sort(key=lambda x: -x["stage1_score"])
    top_pool = stage1_candidates[:STAGE1_POOL]
    below_pool = stage1_candidates[STAGE1_POOL:]

    print(f"\nStage 1 complete. {len(top_pool)} candidates selected for deep analysis.")
    print(f"  ({len(honeypot_scored)} honeypots filtered, "
          f"{len(below_pool)} candidates below pool threshold)")

    # -------------------------------------------------------------------------
    # STAGE 2: Deep Re-Ranking on Top 500 → Final Top 100
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STAGE 2: Deep re-ranking on Top {STAGE1_POOL} candidates...")
    print(f"{'='*60}")

    stage2_results = []

    for entry in tqdm(top_pool, desc="Stage 2 Deep Scoring"):
        c = entry["data"]
        cid = entry["candidate_id"]
        semantic_score = entry["semantic_score"]

        # Full deep scoring with capability relevance, production, hidden talent
        final_score, reasoning, is_stuffing = evaluate_candidate(
            c, semantic_score, deep=True
        )

        stage2_results.append({
            "candidate_id": cid,
            "score": final_score,
            "reasoning": reasoning
        })

    # -------------------------------------------------------------------------
    # 6. Sort & Select Top 100
    # -------------------------------------------------------------------------
    print("\nSorting and selecting Top 100...")
    stage2_results.sort(
        key=lambda x: (-round(x["score"], 4), x["candidate_id"])
    )
    top_100 = stage2_results[:100]

    # -------------------------------------------------------------------------
    # 7. Write CSV
    # -------------------------------------------------------------------------
    print(f"Writing ranked results to {args.out}...")
    with open(args.out, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, item in enumerate(top_100, 1):
            writer.writerow([
                item["candidate_id"],
                rank,
                round(item["score"], 4),
                item["reasoning"]
            ])

    print("Submission CSV written successfully.")

    # -------------------------------------------------------------------------
    # 8. Validate
    # -------------------------------------------------------------------------
    validator_path = os.path.join(
        base_dir,
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "validate_submission.py"
    )
    if os.path.exists(validator_path):
        print(f"Running validator on {args.out}...")
        import subprocess
        res = subprocess.run(
            f'"{sys.executable}" "{validator_path}" "{args.out}"',
            shell=True, capture_output=True, text=True
        )
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
    else:
        print("Warning: validate_submission.py not found.")


if __name__ == "__main__":
    main()
