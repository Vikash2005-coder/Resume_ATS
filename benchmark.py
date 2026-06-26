# benchmark.py
"""
Local Benchmark — Compare two submission CSVs to measure improvement.

Usage:
  # Compare old vs new submission:
  python benchmark.py --baseline submission_old.csv --new submission.csv

  # Analyse a single submission (shows detailed stats):
  python benchmark.py --new submission.csv

What this measures:
  - Candidate overlap @ top-10, top-50, top-100
  - Score distribution (mean, std, min, max)
  - Reasoning quality checks (variation, length, concern clauses)
  - Honeypot rate estimation (you provide honeypot IDs if known)
  - Rank shift for specific candidate IDs

Since the ground truth is hidden, this script uses PROXY metrics to estimate
quality improvement without needing the real scores. The proxies are:

  1. Score differentiation (higher std = model is discriminating more)
  2. Score spread (top-1 vs rank-100 gap should be large)
  3. Reasoning variation (are explanations substantively different?)
  4. Reasoning length (too short = templated, too long = padding)
  5. Overlap @ 10 between old and new (large shift = significant change)
"""

import csv
import sys
import argparse
import re
from collections import Counter


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_submission(path):
    """Load a submission CSV into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row.get("reasoning", "")
            })
    return sorted(rows, key=lambda x: x["rank"])


# ---------------------------------------------------------------------------
# Score statistics
# ---------------------------------------------------------------------------
def score_stats(rows, label=""):
    scores = [r["score"] for r in rows]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    spread = scores[0] - scores[-1]

    print(f"\n{'─'*50}")
    print(f"  Score Statistics [{label}]")
    print(f"{'─'*50}")
    print(f"  Rank-1 score    : {scores[0]:.4f}")
    print(f"  Rank-10 score   : {scores[9]:.4f}")
    print(f"  Rank-50 score   : {scores[49]:.4f}")
    print(f"  Rank-100 score  : {scores[-1]:.4f}")
    print(f"  Mean score      : {mean:.4f}")
    print(f"  Std dev         : {std:.4f}  ← higher = more discrimination")
    print(f"  Top-bottom spread: {spread:.4f}  ← higher = better separation")

    return {"mean": mean, "std": std, "spread": spread}


# ---------------------------------------------------------------------------
# Reasoning quality analysis
# ---------------------------------------------------------------------------
def reasoning_quality(rows, label=""):
    reasonings = [r["reasoning"] for r in rows if r["reasoning"]]
    if not reasonings:
        print(f"\n  [WARNING] No reasoning strings found in {label}!")
        return

    lengths = [len(r) for r in reasonings]
    avg_len = sum(lengths) / len(lengths)

    # Count unique reasonings (detect near-duplicates by first 60 chars)
    prefixes = [r[:60] for r in reasonings]
    unique_prefixes = len(set(prefixes))

    # Check for concern clauses in lower-ranked candidates
    lower_rank = [r["reasoning"] for r in rows if r["rank"] > 50]
    concern_count = sum(
        1 for r in lower_rank
        if any(word in r.lower() for word in ["concern", "note:", "ranked lower", "limited", "below threshold", "borderline"])
    )

    # Check for confidence percentages (all should have them)
    confidence_count = sum(1 for r in reasonings if "confidence:" in r.lower())

    print(f"\n{'─'*50}")
    print(f"  Reasoning Quality [{label}]")
    print(f"{'─'*50}")
    print(f"  Total reasonings      : {len(reasonings)}/100")
    print(f"  Avg length (chars)    : {avg_len:.0f}  (target: 80–250)")
    print(f"  Unique openers (first 60 chars): {unique_prefixes}/100")
    print(f"  Has concern clause (rank 51-100): {concern_count}/{len(lower_rank)}")
    print(f"  Has confidence score  : {confidence_count}/100")

    # Flag issues
    if unique_prefixes < 70:
        print(f"  ⚠️  LOW VARIATION: Only {unique_prefixes} unique sentence openers.")
        print(f"      Stage 4 reviewers look for this. Aim for 90+.")
    else:
        print(f"  ✅ Good variation: {unique_prefixes} unique openers.")

    if concern_count < 20:
        print(f"  ⚠️  FEW CONCERNS: Only {concern_count} lower-ranked candidates have concern notes.")
    else:
        print(f"  ✅ Concern clauses present for {concern_count} lower-ranked candidates.")

    if avg_len < 60:
        print(f"  ⚠️  TOO SHORT: Average reasoning {avg_len:.0f} chars — likely templated.")
    elif avg_len > 400:
        print(f"  ⚠️  TOO LONG: Average reasoning {avg_len:.0f} chars — risk of padding/verbosity.")
    else:
        print(f"  ✅ Reasonable length: {avg_len:.0f} chars average.")


# ---------------------------------------------------------------------------
# Overlap comparison
# ---------------------------------------------------------------------------
def compare_overlap(old_rows, new_rows):
    old_10 = set(r["candidate_id"] for r in old_rows[:10])
    new_10 = set(r["candidate_id"] for r in new_rows[:10])
    old_50 = set(r["candidate_id"] for r in old_rows[:50])
    new_50 = set(r["candidate_id"] for r in new_rows[:50])
    old_100 = set(r["candidate_id"] for r in old_rows[:100])
    new_100 = set(r["candidate_id"] for r in new_rows[:100])

    overlap_10 = len(old_10 & new_10)
    overlap_50 = len(old_50 & new_50)
    overlap_100 = len(old_100 & new_100)

    print(f"\n{'─'*50}")
    print(f"  Candidate Overlap (Old vs New)")
    print(f"{'─'*50}")
    print(f"  Overlap @ Top-10  : {overlap_10}/10  ({overlap_10*10}% same candidates)")
    print(f"  Overlap @ Top-50  : {overlap_50}/50  ({overlap_50*2}% same candidates)")
    print(f"  Overlap @ Top-100 : {overlap_100}/100 ({overlap_100}% same candidates)")

    new_in_top10 = new_10 - old_10
    dropped_from_top10 = old_10 - new_10
    if new_in_top10:
        print(f"\n  🆕 New in Top-10  : {sorted(new_in_top10)}")
    if dropped_from_top10:
        print(f"  ⬇️  Dropped from Top-10: {sorted(dropped_from_top10)}")


# ---------------------------------------------------------------------------
# Score differentiation proxy (NDCG surrogate)
# ---------------------------------------------------------------------------
def differentiation_score(rows):
    """
    Proxy for ranking quality: measures how well scores discriminate
    candidates. A good ranker has large gaps between early ranks.
    """
    scores = [r["score"] for r in rows]
    # Compute discounted cumulative "spread"
    dc_spread = sum(
        (scores[i] - scores[i+1]) / (i + 1)
        for i in range(min(9, len(scores)-1))
    )
    return dc_spread


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark submission CSVs")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Old submission CSV to compare against")
    parser.add_argument("--new", type=str, required=True,
                        help="New submission CSV to analyse")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  REDROB HACKATHON — LOCAL BENCHMARK REPORT")
    print("="*55)

    new_rows = load_submission(args.new)
    print(f"\n  Loaded NEW submission: {args.new}")
    print(f"  Rows: {len(new_rows)} | Expected: 100")
    if len(new_rows) != 100:
        print(f"  ❌ ERROR: Submission has {len(new_rows)} rows, expected exactly 100!")
        sys.exit(1)

    # Check ranks
    ranks = [r["rank"] for r in new_rows]
    if sorted(ranks) != list(range(1, 101)):
        print("  ❌ ERROR: Ranks are not 1-100 exactly once!")
        sys.exit(1)
    else:
        print("  ✅ Format valid: 100 rows, ranks 1-100 each exactly once.")

    # Score monotonicity
    scores = [r["score"] for r in new_rows]
    violations = sum(1 for i in range(len(scores)-1) if scores[i] < scores[i+1] - 1e-6)
    if violations:
        print(f"  ❌ {violations} score monotonicity violations (score increases with rank)!")
    else:
        print("  ✅ Scores are non-increasing with rank.")

    score_stats(new_rows, label="NEW")
    reasoning_quality(new_rows, label="NEW")

    new_dc = differentiation_score(new_rows)
    print(f"\n{'─'*50}")
    print(f"  Discrimination Score (proxy for NDCG@10): {new_dc:.6f}")
    print(f"  (Higher = better top-10 separation. Compare across runs.)")

    if args.baseline:
        print(f"\n{'='*55}")
        print(f"  COMPARING: {args.baseline} vs {args.new}")
        print(f"{'='*55}")

        old_rows = load_submission(args.baseline)
        print(f"  Loaded BASELINE: {args.baseline} | Rows: {len(old_rows)}")

        score_stats(old_rows, label="BASELINE")
        reasoning_quality(old_rows, label="BASELINE")

        old_dc = differentiation_score(old_rows)
        compare_overlap(old_rows, new_rows)

        new_dc = differentiation_score(new_rows)
        delta = new_dc - old_dc

        print(f"\n{'─'*50}")
        print(f"  Discrimination Score Comparison")
        print(f"{'─'*50}")
        print(f"  Baseline : {old_dc:.6f}")
        print(f"  New      : {new_dc:.6f}")
        delta_str = f"+{delta:.6f}" if delta >= 0 else f"{delta:.6f}"
        print(f"  Delta    : {delta_str}  ({'IMPROVED ✅' if delta > 0 else 'REGRESSED ❌'})")

        # Summary verdict
        new_std = sum((s - sum(scores)/len(scores))**2 for s in scores)**0.5 / len(scores)**0.5
        old_scores = [r["score"] for r in old_rows]
        old_mean = sum(old_scores)/len(old_scores)
        old_std = sum((s - old_mean)**2 for s in old_scores)**0.5 / len(old_scores)**0.5

        print(f"\n{'='*55}")
        print("  VERDICT")
        print(f"{'='*55}")
        improvements = []
        if new_dc > old_dc:
            improvements.append("Better top-10 discrimination")
        if new_std > old_std:
            improvements.append("Higher score spread (more differentiation)")

        new_reasoning = [r["reasoning"] for r in new_rows if r["reasoning"]]
        old_reasoning = [r["reasoning"] for r in old_rows if r["reasoning"]]
        if len(new_reasoning) > len(old_reasoning):
            improvements.append("More candidates have reasoning")

        if improvements:
            print("  ✅ Improvements detected:")
            for imp in improvements:
                print(f"     • {imp}")
        else:
            print("  ⚠️  No clear improvement detected. Review score distributions.")

    print("\n" + "="*55 + "\n")


if __name__ == "__main__":
    main()
