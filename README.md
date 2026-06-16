# Intelligent Candidate Discovery & Ranking Engine
**Redrob AI Hackathon — Track 1: Data & AI Challenge**

An optimized, CPU-only candidate ranking engine that evaluates **100,000 candidates** against a Senior AI Engineer Job Description (JD). The system implements hybrid dense-sparse scoring, active honeypot detection, keyword stuffing suppression, and automated submission validation.

---

## System Architecture

```text
               Offline Pre-Computation (embed.py)
 [ candidates.jsonl ] ---> Generate Career Text ---> Local Transformer ---> [ candidate_embeddings.npy ]
                                                                                   |
                                                                                   v
               Runtime Execution (rank.py - Limit: 5 Min)                          |
 [ job_description.md ] ---> Encode JD ---> Cosine Similarity Match <---------------+
                                 |
                                 +---> [ honeypot_detector.py ] (Exclude fakes)
                                 |
                                 +---> [ scorer.py ] (Hierarchical Scorer & Stuffing Check)
                                 |
                                 v
                         [ team_xxx.csv ] ---> [ validate_submission.py ] (Auto-validate)
```

### Key Components

1.  **Honeypot Detector ([honeypot_detector.py](file:///e:/Resume%20ATS/honeypot_detector.py))**: Implements hard disqualification checks (Flags `is_honeypot = True` and overrides score to `0.0`) to avoid honeypots (disqualification threshold: $>10\%$ fakes in top 100). Validates:
    *   *Skill-Duration Trap*: Expert technical skills listed with `duration_months == 0`.
    *   *Experience Mismatch*: Stated profile experience vs cumulative duration of roles (margin $>5$ years).
    *   *Timeline Integrity*: Future roles ($>2026$) or start date after end date.
    *   *Chronology mismatch*: Senior/professional roles starting $>4$ years before graduation.
    *   *Skill Stuffing*: Candidates with $<3$ years of experience claiming $\ge 10$ expert skills.
2.  **Semantic Career Representation ([scorer.py](file:///e:/Resume%20ATS/scorer.py))**: Uses local sentence-transformers embeddings (`all-MiniLM-L6-v2`) to capture context in the candidate summary, headline, and career history descriptions. It matches semantic intent (e.g. building search engines, matching, or recommendation pipelines) rather than simple keyword matches.
3.  **Concept Skill Matcher ([skills_config.py](file:///e:/Resume%20ATS/skills_config.py))**: Translates concrete tools (e.g., *Milvus*, *Pinecone*, *LoRA*, *PEFT*, *BGE*) to high-level JD categories (Vector Databases, LLMs, Retrieval, Evaluation), applying a **Trust Factor** to discount skills with zero duration or low endorsements.
4.  **Behavioral Signal Modulator ([scorer.py](file:///e:/Resume%20ATS/scorer.py))**: Evaluates active availability (notice period $\le 30$ days = 1.0, $>90$ days = 0.1), login recency (under 30 days = 1.0, $>6$ months = 0.1), and response rate metrics.
5.  **Keyword-Stuffing Protection**: Computes a penalty multiplier (0.5) for buzzword-heavy profiles that lack semantic career alignment (low career semantic score).
6.  **Tie-Breaker & Format Compliance**: Breaks score ties alphabetically by `candidate_id` ascending, outputting exactly 100 rows matching the column structure `candidate_id,rank,score,reasoning`.

---

## Final Scoring Weight Distribution

The candidate score is calculated using the following hierarchical weights:

*   **Career Semantic Score**: 25% (Dense embedding similarity between profile text and JD)
*   **Concept Skill Score**: 20% (Overlap in categories, weighted by trust factor)
*   **Title Score**: 15% (Tech title alignment, incorporating Tier-5 generic title upgrades)
*   **Experience Score**: 15% (Soft penalty fit for the target 5-9 years experience)
*   **Behavioral Score**: 15% (Platform engagement, github contributions, and location fit as a 5% optimization)
*   **Availability Score**: 10% (Stated notice period and open-to-work flag)

---

## Installation & Setup

Ensure you are using **Python 3.9+** and install the project requirements:
```bash
pip install -r requirements.txt
```

---

## Execution Instructions

The system is split into two phases to fit the 5-minute runtime CPU constraint:

### 1. Offline Pre-Computation (Runs Once)
Generates and caches candidate embeddings. This step uses multi-threading and runs in the background.
```bash
python embed.py
```
*Note: To run a fast test execution on a subset (e.g., first 100 candidates), run:*
```bash
python embed.py --limit=100
```

### 2. Runtime Scoring & Ranking
Computes the final scores, applies detectors, sorts, tie-breaks, and writes the ranked CSV. It automatically runs the submission validator on completion.
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
*Note: If `rank.py` encounters candidate IDs that are missing from the precomputed embeddings (e.g., when evaluated on a smaller sandbox sample or a custom test set), it automatically falls back to **on-the-fly dynamic embedding generation**, ensuring the script never fails.*

---

## Verification Results
When tested on a 100-candidate sample, the output was successfully validated:
```text
Loading Job Description...
Loaded Job Description (Length: 9564 chars)
Loading Sentence-Transformers model...
Encoding Job Description...
Loading precomputed candidate embeddings...
Successfully loaded 100 candidate embeddings.
Processing and scoring candidates...
Sorting and selecting Top 100...
Writing ranked results to ./submission.csv...
Running validator...
Submission is valid.
```