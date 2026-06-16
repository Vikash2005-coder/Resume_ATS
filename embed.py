# embed.py
"""
Offline script to generate dense text embeddings for 100,000 candidates
using the sentence-transformers/all-MiniLM-L6-v2 model via transformers.
Saves the results in candidate_embeddings.npy.
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Set PyTorch to use all available CPU threads for inference speedup
torch.set_num_threads(os.cpu_count() or 4)

CANDIDATES_PATH = r"e:\Resume ATS\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
OUTPUT_VECTORS_PATH = r"e:\Resume ATS\candidate_embeddings.npy"
OUTPUT_IDS_PATH = r"e:\Resume ATS\candidate_ids.json"

def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings from token embeddings."""
    token_embeddings = model_output[0] # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
    
    # Combine all parts
    full_text = f"{current_title} - {headline}. {summary}. Experience: {career_history_str}"
    
    # Limit text length to avoid memory or processing bottlenecks
    return full_text[:1200]

def main():
    print("Loading tokenizer and model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval() # Set to evaluation mode
    
    import sys
    limit = None
    for arg in sys.argv:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])

    print(f"Reading candidates from {CANDIDATES_PATH}...")
    candidate_ids = []
    texts = []
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            candidate_ids.append(c["candidate_id"])
            texts.append(construct_career_text(c))
            if limit and len(candidate_ids) >= limit:
                break
            
    num_candidates = len(candidate_ids)
    print(f"Loaded {num_candidates} candidate texts. Beginning embedding computation...")
    
    batch_size = 128
    embeddings_list = []
    
    # Run embedding extraction in batches
    with torch.no_grad():
        for i in tqdm(range(0, num_candidates, batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encoded_input = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=256, 
                return_tensors='pt'
            )
            
            # Compute token embeddings
            model_output = model(**encoded_input)
            
            # Perform mean pooling to get sentence embeddings
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings (so cosine similarity is just dot product)
            # Normalization formula: v = v / l2_norm(v)
            norms = torch.norm(batch_embeddings, p=2, dim=1, keepdim=True)
            batch_embeddings_normalized = batch_embeddings / torch.clamp(norms, min=1e-9)
            
            embeddings_list.append(batch_embeddings_normalized.cpu().numpy())
            
    # Concatenate all batch embeddings
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Embedding shape: {all_embeddings.shape}")
    
    # Save embeddings and ID mapping
    print(f"Saving embeddings to {OUTPUT_VECTORS_PATH}...")
    np.save(OUTPUT_VECTORS_PATH, all_embeddings)
    
    print(f"Saving candidate IDs to {OUTPUT_IDS_PATH}...")
    with open(OUTPUT_IDS_PATH, "w", encoding="utf-8") as out:
        json.dump(candidate_ids, out)
        
    print("Embedding pre-computation complete!")

if __name__ == '__main__':
    main()
