# scorer.py
"""
Candidate Intelligence Engine - Scoring Module (Phase 2)
Implements the 7-component hierarchical scoring formula:
  25% Career Semantic Score (70% cosine + 30% capability relevance)
  20% Concept Skill Score
  15% Experience Score
  15% Behavioral Score
  10% Availability Score
  10% Production Readiness Score
   5% Hidden Talent Score (generic titles only)

Plus: Density-based keyword stuffing detection, dynamic evidence-based
explanations, and a candidate confidence percentage.
"""

import math
from datetime import datetime
import dateutil.parser
from skills_config import (
    CORE_CONCEPTS, AI_TITLE_KEYWORDS, GENERIC_TECH_TITLES,
    IRRELEVANT_TITLES, TIER5_CAREER_KEYWORDS,
    CAPABILITIES, PRODUCTION_KEYWORDS
)


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------
def parse_date_local(date_str):
    if not date_str:
        return None
    try:
        return dateutil.parser.parse(date_str)
    except Exception:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 1. Capability-Based Career Relevance Score
# ---------------------------------------------------------------------------
def calculate_career_relevance_score(career_history):
    """
    Score based on how many of the 4 recruiter-facing capability groups
    (retrieval, ranking, recommendation, matching) are evidenced in the
    candidate's career history descriptions.

    Returns a float in [0, 1].
    """
    # Concatenate all role descriptions into one lowercased blob
    combined = " ".join(
        role.get("description", "") for role in career_history
    ).lower()

    capabilities_found = 0
    total_capabilities = len(CAPABILITIES)

    for cap_keywords in CAPABILITIES.values():
        if any(kw in combined for kw in cap_keywords):
            capabilities_found += 1

    return capabilities_found / total_capabilities if total_capabilities > 0 else 0.0


# ---------------------------------------------------------------------------
# 2. Production Readiness Score (10%)
# ---------------------------------------------------------------------------
def calculate_production_score(career_history):
    """
    Evaluates production deployment maturity across 4 continuous dimensions.
    Each dimension score = unique_keyword_matches / total_keywords_in_dimension.
    Final score = average of 4 dimension scores.
    """
    combined = " ".join(
        role.get("description", "") for role in career_history
    ).lower()

    dimension_scores = []
    for dim_keywords in PRODUCTION_KEYWORDS.values():
        unique_matches = sum(1 for kw in dim_keywords if kw in combined)
        dimension_scores.append(min(1.0, unique_matches / len(dim_keywords)))

    return sum(dimension_scores) / len(dimension_scores) if dimension_scores else 0.0


# ---------------------------------------------------------------------------
# 3. Skill Trust Factor
# ---------------------------------------------------------------------------
def calculate_trust_factor(duration_months, endorsements):
    """Calculates a skill trust multiplier to discount keyword stuffers."""
    duration_factor = min(1.0, duration_months / 12.0)
    endorsement_factor = 0.5 + 0.5 * min(1.0, endorsements / 10.0)
    return duration_factor * endorsement_factor


# ---------------------------------------------------------------------------
# 4. Concept Skill Score (20%)
# ---------------------------------------------------------------------------
def calculate_skills_score(skills):
    """
    Computes a concept-based semantic skill score.
    Groups candidate skills into JD core requirements and weights by
    duration/endorsements/proficiency.
    """
    category_scores = {cat: 0.0 for cat in CORE_CONCEPTS}

    for skill in skills:
        name = skill.get("name", "").lower()
        proficiency_str = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        prof_mult = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5}.get(
            proficiency_str, 0.2
        )
        trust = calculate_trust_factor(duration, endorsements)
        skill_val = prof_mult * trust

        for cat, config in CORE_CONCEPTS.items():
            if name in config["keywords"] or any(kw in name for kw in config["keywords"]):
                category_scores[cat] = min(1.0, category_scores[cat] + skill_val)

    total_score = sum(
        category_scores[cat] * CORE_CONCEPTS[cat]["weight"]
        for cat in CORE_CONCEPTS
    )
    total_weight = sum(c["weight"] for c in CORE_CONCEPTS.values())
    return total_score / total_weight if total_weight > 0 else 0.0


# ---------------------------------------------------------------------------
# 5. Experience Score (15%)
# ---------------------------------------------------------------------------
def calculate_experience_score(years_exp):
    """Fits years of experience against the JD's 5-9 years target."""
    if 5.0 <= years_exp <= 9.0:
        return 1.0
    elif 4.0 <= years_exp < 5.0 or 9.0 < years_exp <= 11.0:
        return 0.8
    elif 3.0 <= years_exp < 4.0 or 11.0 < years_exp <= 13.0:
        return 0.5
    elif 2.0 <= years_exp < 3.0 or 13.0 < years_exp <= 15.0:
        return 0.3
    else:
        return 0.1


# ---------------------------------------------------------------------------
# 6. Behavioral Score (15%)
# ---------------------------------------------------------------------------
def calculate_behavioral_score(signals):
    """
    Combines location preference, platform activity, and recruiter engagement.
    Schema-aware: missing fields default to neutral values.
    """
    loc = signals.get("location", "").lower()
    relocate = signals.get("willing_to_relocate", False)

    is_preferred = any(city in loc for city in [
        "noida", "pune", "ncr", "delhi ncr", "gurgaon", "ghaziabad", "faridabad"
    ])
    is_tier1 = any(city in loc for city in [
        "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata"
    ])

    if is_preferred:
        location_score = 1.0
    elif is_tier1:
        location_score = 0.9
    else:
        location_score = 0.9 if relocate else 0.8

    # Activity score (defensive - default neutral if field missing)
    last_act = parse_date_local(signals.get("last_active_date"))
    if last_act:
        days_inactive = (datetime(2026, 6, 17) - last_act).days
        if days_inactive <= 30:
            activity_score = 1.0
        elif days_inactive <= 90:
            activity_score = 0.8
        elif days_inactive <= 180:
            activity_score = 0.5
        else:
            activity_score = 0.1
    else:
        activity_score = 0.5

    # Engagement score (all fields default to neutral if missing)
    resp_rate = signals.get("recruiter_response_rate", 0.5)
    inter_rate = signals.get("interview_completion_rate", 0.5)
    github_score = signals.get("github_activity_score", -1)
    github_val = github_score / 100.0 if github_score >= 0 else 0.5

    # Safely clamp to [0, 1]
    resp_rate = max(0.0, min(1.0, float(resp_rate)))
    inter_rate = max(0.0, min(1.0, float(inter_rate)))
    github_val = max(0.0, min(1.0, float(github_val)))

    engagement_score = (resp_rate + inter_rate + github_val) / 3.0
    return (location_score + activity_score + engagement_score) / 3.0


# ---------------------------------------------------------------------------
# 7. Availability Score (10%)
# ---------------------------------------------------------------------------
def calculate_availability_score(signals):
    """Calculates notice period suitability."""
    notice = signals.get("notice_period_days", 60)
    open_to_work = signals.get("open_to_work_flag", False)

    if notice <= 30:
        base = 1.0
    elif notice <= 60:
        base = 0.8
    elif notice <= 90:
        base = 0.5
    else:
        base = 0.1

    multiplier = 1.1 if open_to_work else 0.9
    return min(1.0, max(0.0, base * multiplier))


# ---------------------------------------------------------------------------
# 8. Hidden Talent Score (5%) — Generic titles only
# ---------------------------------------------------------------------------
def calculate_hidden_talent_score(profile, career_history, career_relevance_score):
    """
    Rewards candidates with generic (non-AI) titles who have demonstrable
    retrieval/ranking/search experience in their career history.
    Returns 0.0 for explicit AI/ML titles or irrelevant titles.
    """
    curr_title = profile.get("current_title", "").lower()

    is_generic = any(kw in curr_title for kw in GENERIC_TECH_TITLES)
    is_ai = any(kw in curr_title for kw in AI_TITLE_KEYWORDS)
    is_irrelevant = any(kw in curr_title for kw in IRRELEVANT_TITLES)

    if is_irrelevant or is_ai:
        return 0.0

    generic_title_factor = 1.0 if is_generic else 0.0
    return career_relevance_score * generic_title_factor


# ---------------------------------------------------------------------------
# 9. Density-Based Keyword Stuffing Detection
# ---------------------------------------------------------------------------
def detect_keyword_stuffing(skills, semantic_score, career_history):
    """
    Detects keyword-heavy profiles with weak actual work evidence.
    Uses density ratio rather than raw skill counts.
    """
    total_skills = len(skills)
    if total_skills == 0:
        return False

    high_prof_count = sum(
        1 for s in skills
        if s.get("proficiency", "").lower() in ["expert", "advanced"]
    )
    keyword_density = high_prof_count / total_skills

    # Check if career history provides actual retrieval/AI evidence
    combined = " ".join(
        role.get("description", "") for role in career_history
    ).lower()
    evidence_keywords = [
        "retrieval", "ranking", "recommendation", "vector", "embedding",
        "search", "nlp", "machine learning", "neural", "model"
    ]
    has_evidence = any(kw in combined for kw in evidence_keywords)

    # Penalise if: high skill density + low semantic relevance + no career evidence
    if keyword_density >= 0.70 and semantic_score < 0.45 and not has_evidence:
        return True
    # Legacy fallback for extremely stuffed profiles
    if high_prof_count >= 10 and semantic_score < 0.40:
        return True

    return False


# ---------------------------------------------------------------------------
# 10. Confidence Score Helper
# ---------------------------------------------------------------------------
def calculate_confidence(scores_dict):
    """
    Computes a confidence percentage (50–99%) based on agreement across
    core scoring signals. High agreement = high confidence.
    """
    values = list(scores_dict.values())
    if not values:
        return 75
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_dev = math.sqrt(variance)
    confidence = (0.70 * mean_val + 0.30 * (1.0 - std_dev)) * 100
    return int(min(99, max(50, confidence)))


# ---------------------------------------------------------------------------
# 11. Dynamic Evidence-Based Reasoning Generator
# ---------------------------------------------------------------------------
def generate_reasoning(candidate, semantic_score, skills_score, career_relevance_score,
                        production_score, hidden_talent_score, final_score):
    """
    Generates a unique, evidence-based recruiter explanation for this candidate.
    """
    prof = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    years_exp = prof.get("years_of_experience", 0.0)
    current_title = prof.get("current_title", "Engineer")
    notice = signals.get("notice_period_days", 60)
    resp_rate = signals.get("recruiter_response_rate", 0.5)
    github_score = signals.get("github_activity_score", -1)

    skill_names = [s.get("name", "").lower() for s in skills]
    combined_career = " ".join(
        role.get("description", "") for role in career
    ).lower()

    # --- Build strength clauses from actual evidence ---
    strengths = []

    # Check capabilities found in career descriptions
    for cap_name, cap_keywords in CAPABILITIES.items():
        if any(kw in combined_career for kw in cap_keywords):
            strengths.append(cap_name.replace("_", " "))

    # Check skills list for additional hard-skill evidence
    skill_evidence = []
    if any(kw in s for s in skill_names for kw in CORE_CONCEPTS["vector_databases"]["keywords"]):
        skill_evidence.append("vector database expertise")
    if any(kw in s for s in skill_names for kw in CORE_CONCEPTS["llm_dl"]["keywords"]):
        skill_evidence.append("LLM fine-tuning")

    # Merge strengths from both sources, deduplicate
    all_strengths = list(dict.fromkeys(strengths + skill_evidence))

    if not all_strengths:
        all_strengths = ["applied machine learning", "data engineering"]

    strengths_str = ", ".join(all_strengths[:3])

    # --- Production clause ---
    production_clause = ""
    if production_score >= 0.5:
        production_clause = " with strong production deployment experience"
    elif production_score >= 0.25:
        production_clause = " with some production deployment exposure"

    # --- Engagement clause ---
    if resp_rate > 0.6:
        engagement_clause = "high recruiter engagement"
    elif github_score >= 50:
        engagement_clause = "strong open-source activity"
    else:
        engagement_clause = "good platform engagement"

    # --- Hidden talent clause ---
    hidden_clause = ""
    curr_title_lower = current_title.lower()
    if hidden_talent_score > 0.3 and any(kw in curr_title_lower for kw in GENERIC_TECH_TITLES):
        hidden_clause = " Hidden talent: strong retrieval/search background despite generic title."

    # --- Notice/availability clause ---
    availability_clause = ""
    if notice > 90:
        availability_clause = f" A long notice period of {notice} days is noted as a concern."
    elif notice >= 60:
        availability_clause = " A 60-day notice period is noted as a minor concern."

    # --- Confidence score ---
    confidence = calculate_confidence({
        "semantic": semantic_score,
        "skills": skills_score,
        "career_relevance": career_relevance_score,
        "production": production_score
    })

    # --- Assemble final reasoning ---
    reasoning = (
        f"Demonstrated capability in {strengths_str}{production_clause}, "
        f"with {years_exp:.1f} years of engineering experience as a {current_title} "
        f"and {engagement_clause}.{hidden_clause}{availability_clause} "
        f"(Confidence: {confidence}%)"
    )
    return reasoning


# ---------------------------------------------------------------------------
# 12. Master evaluate_candidate — Full Scoring
# ---------------------------------------------------------------------------
def evaluate_candidate(candidate, semantic_score, deep=False):
    """
    Evaluates a candidate profile against all scoring criteria.

    Args:
        candidate: The full candidate JSON object.
        semantic_score: Pre-computed cosine similarity score (float).
        deep: If True, applies career relevance, production, and hidden talent
              scoring (Stage 2 only). If False, uses fast approximations (Stage 1).

    Returns:
        final_score (float), reasoning (str), is_stuffing (bool)
    """
    prof = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    # Basic scores (always computed)
    s_skills = calculate_skills_score(skills)
    s_experience = calculate_experience_score(prof.get("years_of_experience", 0.0))

    # Schema-aware behavioral scoring
    behavior_dict = dict(signals)
    behavior_dict["location"] = prof.get("location", "")
    behavior_dict["country"] = prof.get("country", "")
    s_behavior = calculate_behavioral_score(behavior_dict)
    s_availability = calculate_availability_score(signals)

    # Deep Stage 2 scores (capability-based, expensive text scanning)
    if deep:
        career_relevance = calculate_career_relevance_score(career)
        s_production = calculate_production_score(career)
        s_hidden = calculate_hidden_talent_score(prof, career, career_relevance)

        # Blend cosine similarity with capability relevance inside semantic score
        blended_semantic = 0.70 * semantic_score + 0.30 * career_relevance

        # Keyword stuffing detection (density-based)
        is_stuffing = detect_keyword_stuffing(skills, semantic_score, career)

        # 7-component formula
        raw_score = (
            0.25 * blended_semantic +
            0.20 * s_skills +
            0.15 * s_experience +
            0.15 * s_behavior +
            0.10 * s_availability +
            0.10 * s_production +
            0.05 * s_hidden
        )

        final_score = raw_score * 0.5 if is_stuffing else raw_score

        reasoning = generate_reasoning(
            candidate, blended_semantic, s_skills, career_relevance,
            s_production, s_hidden, final_score
        )
    else:
        # Stage 1 fast approximation (no text scanning)
        career_relevance = 0.0
        s_production = 0.0
        s_hidden = 0.0
        is_stuffing = False

        # Simplified fast score matching original formula
        raw_score = (
            0.25 * semantic_score +
            0.20 * s_skills +
            0.15 * s_experience +
            0.15 * s_behavior +
            0.10 * s_availability
        )
        final_score = raw_score
        reasoning = ""

    return final_score, reasoning, is_stuffing
