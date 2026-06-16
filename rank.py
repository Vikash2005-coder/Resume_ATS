# rank.py
"""
Candidate Ranking CLI for the Redrob AI Hackathon.
Streams candidates, dynamically computes or retrieves embeddings,
runs the multi-signal scorer, resolves ties, and writes the final ranked CSV.
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

# Add local path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from honeypot_detector import is_honeypot
from scorer import evaluate_candidate

# Set PyTorch thread count
torch.set_num_threads(os.cpu_count() or 4)

# Load docx utility if needed
def get_docx_text(path):
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return ""

def load_job_description(base_path):
    """Loads the job description text from markdown or word format."""
    # Check common locations
    jd_paths = [
        os.path.join(base_path, "job_description.md"),
        os.path.join(base_path, "job_description.docx"),
        os.path.join(base_path, "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "job_description.docx")
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
                    
    # Fallback default JD summary if files are completely missing
    print("Warning: Job description file not found. Using default embedded JD properties.")
    return (
        "Senior AI Engineer - Founding Team. Redrob AI. Series A talent intelligence platform. "
        "Must have experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5), "
        "vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). "
        "Strong Python. Experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP). "
        "LLM fine-tuning experience (LoRA, QLoRA, PEFT). Learning-to-rank models. Pune/Noida location."
    )

def construct_career_text(candidate):
    """Constructs a comprehensive text summarizing the candidate's profile and experience."""
    prof = candidate.get("profile", {})
    summary = prof.get("summary", "")
    headline = prof.get("headline", "")
    current_title = prof.get("current_title", "")
    
    roles = candidate.get("career_history", [])
    roles_text = []
    for role in roles:
        title = role.get("title", "")
        desc = role.get("description", "")
        roles_text.append(f"{title}: {desc}")
        
    career_history_str = " | ".join(roles_text)
    full_text = f"{current_title} - {headline}. {summary}. Experience: {career_history_str}"
    return full_text[:1200]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    parser = argparse.ArgumentParser(description="Rank candidates against the Senior AI Engineer JD.")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", help="Path to candidates JSONL dataset")
    parser.add_argument("--out", type=str, default="./submission.csv", help="Path to write the ranked output CSV")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Job Description
    print("Loading Job Description...")
    jd_text = load_job_description(base_dir)
    print(f"Loaded Job Description (Length: {len(jd_text)} chars)")
    
    # 2. Load Transformer Model
    print("Loading Sentence-Transformers model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # 3. Generate JD Embedding
    print("Encoding Job Description...")
    with torch.no_grad():
        encoded_jd = tokenizer(jd_text[:1500], padding=True, truncation=True, max_length=256, return_tensors='pt')
        jd_output = model(**encoded_jd)
        jd_embedding = mean_pooling(jd_output, encoded_jd['attention_mask'])[0]
        # Normalize
        jd_embedding = jd_embedding / torch.norm(jd_embedding, p=2)
        jd_vec = jd_embedding.cpu().numpy()
        
    # 4. Load Pre-computed Candidate Embeddings
    embeddings_lookup = {}
    vectors_path = os.path.join(base_dir, "candidate_embeddings.npy")
    ids_path = os.path.join(base_dir, "candidate_ids.json")
    
    if os.path.exists(vectors_path) and os.path.exists(ids_path):
        print(f"Loading precomputed candidate embeddings from {vectors_path}...")
        try:
            vectors = np.load(vectors_path)
            with open(ids_path, "r", encoding="utf-8") as f:
                cids = json.load(f)
            if len(cids) == len(vectors):
                for cid, vec in zip(cids, vectors):
                    embeddings_lookup[cid] = vec
                print(f"Successfully loaded {len(embeddings_lookup)} candidate embeddings.")
            else:
                print("Warning: Candidate ID count mismatch in precomputed vectors. Falling back to dynamic encoding.")
        except Exception as e:
            print(f"Warning: Failed to load precomputed embeddings: {e}. Falling back to dynamic encoding.")
    else:
        print("Precomputed embeddings not found. All embeddings will be generated dynamically.")

    # 5. Process Candidates
    print(f"Processing and scoring candidates from {args.candidates}...")
    scored_candidates = []
    
    # Check number of lines for progress bar if possible
    num_lines = None
    try:
        with open(args.candidates, "rb") as f:
            num_lines = sum(1 for _ in f)
    except:
        pass
        
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="Scoring Candidates"):
            if not line.strip():
                continue
            c = json.loads(line)
            cid = c["candidate_id"]
            
            # Check Honeypots
            flagged, reasons = is_honeypot(c)
            if flagged:
                # Disqualified - Score is 0.0
                scored_candidates.append({
                    "candidate_id": cid,
                    "score": 0.0,
                    "reasoning": f"Profile flagged during automated credentials audit. Reasons: {', '.join(reasons[:2])}."
                })
                continue
                
            # Retrieve or generate embedding
            if cid in embeddings_lookup:
                c_vec = embeddings_lookup[cid]
            else:
                # Compute on the fly (for sandbox or missing entries)
                c_text = construct_career_text(c)
                with torch.no_grad():
                    encoded_c = tokenizer(c_text, padding=True, truncation=True, max_length=256, return_tensors='pt')
                    c_output = model(**encoded_c)
                    c_emb = mean_pooling(c_output, encoded_c['attention_mask'])[0]
                    c_emb = c_emb / torch.norm(c_emb, p=2)
                    c_vec = c_emb.cpu().numpy()
            
            # Compute semantic score (cosine similarity = dot product since normalized)
            semantic_score = float(np.dot(c_vec, jd_vec))
            
            # Calculate final hierarchical score and reasoning
            final_score, reasoning, is_stuffing = evaluate_candidate(c, semantic_score)
            
            scored_candidates.append({
                "candidate_id": cid,
                "score": final_score,
                "reasoning": reasoning
            })

    # 6. Sort Candidates & Resolve Ties
    print("Sorting and selecting Top 100...")
    # Sorting key: Score descending, then Candidate ID ascending alphanumeric order
    scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    top_100 = scored_candidates[:100]
    
    # 7. Write to CSV
    print(f"Writing ranked results to {args.out}...")
    # Write exactly 100 rows + header
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

    # 8. Run Verification Script
    validator_path = os.path.join(base_dir, "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "validate_submission.py")
    if os.path.exists(validator_path):
        print(f"Running validator on {args.out}...")
        import subprocess
        # Run using Python 3.9 path to ensure correct libraries
        cmd = f'"{sys.executable}" "{validator_path}" "{args.out}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
    else:
        print("Warning: validate_submission.py not found in the bundle path.")

if __name__ == '__main__':
    main()
