# rank.py
"""
Candidate Intelligence Engine - Ranking CLI (Phase 2)
Three-stage ranking architecture:
  Stage 1: Fast retrieval on all 100,000 candidates → Top 1000   [A6: was 500]
  Stage 2: Deep re-ranking with full scoring on Top 1000 → Top 200
  Stage 3: Cross-encoder re-ranking of Top 200 → Final Top 100   [Option C: new]

Changes:
  A1: Uses BAAI/bge-small-en-v1.5 with BGE query prefix for better retrieval
  A2: construct_career_text now weights recent roles with more character budget
  A6: STAGE1_POOL increased from 500 → 1000 for a safer candidate buffer
  A5: Rank passed to evaluate_candidate() for rank-aware reasoning
  Option C: Cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) added as Stage 3
            to re-rank Top 200 candidates — dramatically improves top-10 quality
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

# A6: Increased pool for deeper candidate buffer
STAGE1_POOL = 1000   # was 500

# Option C: Cross-encoder re-ranks the top this many candidates
CROSS_ENCODER_POOL = 200

# A1: BGE requires a prefix on the query (JD) side — NOT on documents (candidates)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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
    """
    A2: Smarter career text — weights recent roles more heavily.
    Recent 2 roles get 300 chars; older roles get 100 chars.
    Skills appended as keyword list.
    """
    prof = candidate.get("profile", {})
    summary = prof.get("summary", "")[:200]
    headline = prof.get("headline", "")[:150]
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
        title = role.get("title", "")
        desc = role.get("description", "")
        char_limit = 300 if i < 2 else 100
        role_texts.append(f"{title}: {desc[:char_limit]}")

    career_str = " | ".join(role_texts)
    skills_text = " ".join(s.get("name", "") for s in candidate.get("skills", [])[:15])

    full_text = (
        f"{current_title}. {headline}. {summary}. "
        f"Skills: {skills_text}. "
        f"Experience: {career_str}"
    )
    return full_text[:1500]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# ---------------------------------------------------------------------------
# Option C: Cross-Encoder Stage 3
# ---------------------------------------------------------------------------
def load_cross_encoder():
    """
    Load the cross-encoder model for Stage 3 re-ranking.
    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 — small, fast, CPU-friendly,
    trained for passage-query relevance scoring.
    Returns (model, tokenizer) or (None, None) if unavailable.
    """
    try:
        from transformers import AutoModelForSequenceClassification
        ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"Loading cross-encoder: {ce_model_name} ...")
        ce_tokenizer = AutoTokenizer.from_pretrained(ce_model_name)
        ce_model = AutoModelForSequenceClassification.from_pretrained(ce_model_name)
        ce_model.eval()
        print("Cross-encoder loaded successfully.")
        return ce_model, ce_tokenizer
    except Exception as e:
        print(f"Warning: Could not load cross-encoder ({e}). Stage 3 will be skipped.")
        return None, None


def cross_encoder_score(ce_model, ce_tokenizer, jd_text, candidates_data, batch_size=16):
    """
    Score (JD, candidate_text) pairs using the cross-encoder.
    Returns a dict of {candidate_id: ce_score}.
    The cross-encoder reads BOTH texts together — much more accurate than
    cosine similarity for determining true relevance.
    """
    jd_snippet = jd_text[:512]  # Cross-encoder input is limited
    results = {}

    ids = [c["candidate_id"] for c in candidates_data]
    texts = [construct_career_text(c["data"]) for c in candidates_data]
    pairs = [[jd_snippet, t[:512]] for t in texts]

    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="Stage 3 Cross-Encoder"):
            batch_pairs = pairs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            encoded = ce_tokenizer(
                [p[0] for p in batch_pairs],
                [p[1] for p in batch_pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            logits = ce_model(**encoded).logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()

            for cid, score in zip(batch_ids, scores):
                results[cid] = float(score)

    return results


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
    parser.add_argument(
        "--no-cross-encoder", action="store_true",
        help="Skip Stage 3 cross-encoder re-ranking (faster but lower quality)"
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
    # 2. Load Bi-Encoder Model (A1: BGE-small)
    # -------------------------------------------------------------------------
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Loading bi-encoder model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # -------------------------------------------------------------------------
    # 3. Encode JD — A1: BGE requires query prefix on the JD side
    # -------------------------------------------------------------------------
    print("Encoding Job Description with BGE query prefix...")
    jd_for_encoding = BGE_QUERY_PREFIX + jd_text[:1500]
    with torch.no_grad():
        encoded_jd = tokenizer(
            jd_for_encoding, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
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
    # STAGE 1: Fast Scoring on ALL 100,000 Candidates → Top 1000 (A6: was 500)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STAGE 1: Fast retrieval across all candidates (pool={STAGE1_POOL})...")
    print(f"{'='*60}")

    stage1_candidates = []
    honeypot_scored = []

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

            # Honeypot check (always run)
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

            # Semantic embedding (use precomputed or compute on-the-fly)
            if cid in embeddings_lookup:
                c_vec = embeddings_lookup[cid]
            else:
                c_text = construct_career_text(c)
                # Note: Document side — no BGE prefix on candidates
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

            fast_score, _, _ = evaluate_candidate(c, semantic_score, deep=False)

            stage1_candidates.append({
                "candidate_id": cid,
                "data": c,
                "semantic_score": semantic_score,
                "stage1_score": fast_score
            })

    stage1_candidates.sort(key=lambda x: -x["stage1_score"])
    top_pool = stage1_candidates[:STAGE1_POOL]
    below_pool = stage1_candidates[STAGE1_POOL:]

    print(f"\nStage 1 complete.")
    print(f"  Top pool selected: {len(top_pool)}")
    print(f"  Honeypots filtered: {len(honeypot_scored)}")
    print(f"  Below threshold: {len(below_pool)}")

    # -------------------------------------------------------------------------
    # STAGE 2: Deep Re-Ranking on Top 1000 → Top 200
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STAGE 2: Deep re-ranking on Top {STAGE1_POOL} candidates...")
    print(f"{'='*60}")

    stage2_results = []

    for entry in tqdm(top_pool, desc="Stage 2 Deep Scoring"):
        c = entry["data"]
        cid = entry["candidate_id"]
        semantic_score = entry["semantic_score"]

        final_score, reasoning, is_stuffing = evaluate_candidate(
            c, semantic_score, deep=True, rank=None  # rank assigned after sorting
        )

        stage2_results.append({
            "candidate_id": cid,
            "data": c,
            "score": final_score,
            "reasoning": reasoning
        })

    # Sort Stage 2 results and pick Top 200 for cross-encoder
    stage2_results.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
    top_ce_pool = stage2_results[:CROSS_ENCODER_POOL]

    print(f"\nStage 2 complete. Top {CROSS_ENCODER_POOL} candidates selected for cross-encoder.")

    # -------------------------------------------------------------------------
    # STAGE 3: Cross-Encoder Re-Ranking of Top 200 → Final ranking
    # Option C: Uses (JD, candidate_text) pairs — much more accurate than cosine
    # -------------------------------------------------------------------------
    use_cross_encoder = not args.no_cross_encoder

    if use_cross_encoder:
        print(f"\n{'='*60}")
        print(f"STAGE 3: Cross-encoder re-ranking of Top {CROSS_ENCODER_POOL}...")
        print(f"{'='*60}")

        ce_model, ce_tokenizer = load_cross_encoder()

        if ce_model is not None:
            ce_scores = cross_encoder_score(ce_model, ce_tokenizer, jd_text, top_ce_pool)

            # Blend: 60% cross-encoder + 40% Stage 2 score for stability
            for entry in top_ce_pool:
                cid = entry["candidate_id"]
                ce_score = ce_scores.get(cid, 0.5)
                entry["final_score"] = 0.60 * ce_score + 0.40 * entry["score"]

            top_ce_pool.sort(
                key=lambda x: (-round(x["final_score"], 4), x["candidate_id"])
            )
            print("Stage 3 complete. Final ranking uses blended cross-encoder + deep score.")
        else:
            # Fall back: use Stage 2 score as final
            for entry in top_ce_pool:
                entry["final_score"] = entry["score"]
            print("Stage 3 skipped (cross-encoder unavailable). Using Stage 2 scores.")
    else:
        print("\nStage 3 skipped (--no-cross-encoder flag). Using Stage 2 scores.")
        for entry in top_ce_pool:
            entry["final_score"] = entry["score"]

    # -------------------------------------------------------------------------
    # Regenerate reasoning with correct rank (A5: rank-aware reasoning)
    # -------------------------------------------------------------------------
    print("\nGenerating rank-aware reasoning for Top 100...")
    top_100 = top_ce_pool[:100]

    for rank_idx, entry in enumerate(top_100, 1):
        c = entry["data"]
        semantic_score = next(
            (e["semantic_score"] for e in top_pool if e["candidate_id"] == entry["candidate_id"]),
            0.5
        )
        _, reasoning, _ = evaluate_candidate(c, semantic_score, deep=True, rank=rank_idx)
        entry["reasoning"] = reasoning

    # -------------------------------------------------------------------------
    # Write CSV
    # -------------------------------------------------------------------------
    print(f"\nWriting ranked results to {args.out}...")
    with open(args.out, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_idx, item in enumerate(top_100, 1):
            writer.writerow([
                item["candidate_id"],
                rank_idx,
                round(item.get("final_score", item["score"]), 4),
                item["reasoning"]
            ])

    print("Submission CSV written successfully.")

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------
    validator_path = os.path.join(
        base_dir,
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "validate_submission.py"
    )
    if os.path.exists(validator_path):
        print(f"\nRunning validator on {args.out}...")
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
