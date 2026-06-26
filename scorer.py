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
   5% Assessment Score (verified platform skill scores — replaces hidden talent)

Plus: Density-based keyword stuffing detection, rank-aware evidence-based
explanations, and a candidate confidence percentage.

CHANGES (v2):
  - A3: skill_assessment_scores now used (verified signal, previously unused)
  - A4: Added saved_by_recruiters_30d, offer_acceptance_rate,
         avg_response_time_hours, applications_submitted_30d,
         profile_completeness_score to behavioral scoring
  - A5: Reasoning is now rank-aware with varied openers, concern clauses
        for lower-ranked candidates, and no fixed template structure
"""

import math
import random
from datetime import datetime
import dateutil.parser
from skills_config import (
    CORE_CONCEPTS, AI_TITLE_KEYWORDS, GENERIC_TECH_TITLES,
    IRRELEVANT_TITLES, TIER5_CAREER_KEYWORDS,
    CAPABILITIES, PRODUCTION_KEYWORDS, TECH_DISPLAY, CAP_ACTIONS
)

# JD-relevant skill keywords for assessment score lookup
JD_ASSESSMENT_KEYWORDS = [
    "machine learning", "python", "nlp", "retrieval", "ranking",
    "deep learning", "transformer", "embedding", "vector", "search",
    "recommendation", "data science", "statistics", "pytorch", "tensorflow"
]


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
    Score based on how many of the 6 recruiter-facing capability groups
    (retrieval, ranking, recommendation, matching, vector_database, llm_finetuning)
    are evidenced in the candidate's career history descriptions.

    Returns a float in [0, 1].
    """
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
    Evaluates production deployment maturity across 4 conceptual dimensions:
    deployment, scale, reliability, engineering.

    Each dimension is binary (1 if ANY keyword is found, 0 if none).
    This avoids overweighting repeated mentions of the same underlying concept.
    Final score = fraction of dimensions evidenced (0.0 – 1.0).
    """
    combined = " ".join(
        role.get("description", "") for role in career_history
    ).lower()

    dimensions_found = sum(
        1 for dim_keywords in PRODUCTION_KEYWORDS.values()
        if any(kw in combined for kw in dim_keywords)
    )
    return dimensions_found / len(PRODUCTION_KEYWORDS)


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
# 6. Behavioral Score (15%) — A4: expanded with 5 additional signals
# ---------------------------------------------------------------------------
def calculate_behavioral_score(signals):
    """
    Combines location preference, platform activity, and recruiter engagement.
    Schema-aware: missing fields default to neutral values.

    A4 additions:
      - saved_by_recruiters_30d  (market demand signal)
      - applications_submitted_30d (active job seeker signal)
      - offer_acceptance_rate     (commitment/reliability signal)
      - avg_response_time_hours   (responsiveness signal)
      - profile_completeness_score (effort/seriousness signal)
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

    # --- Core engagement signals (original) ---
    resp_rate = float(signals.get("recruiter_response_rate", 0.5))
    inter_rate = float(signals.get("interview_completion_rate", 0.5))
    github_score = signals.get("github_activity_score", -1)
    github_val = github_score / 100.0 if github_score >= 0 else 0.5

    # Clamp to [0,1]
    resp_rate = max(0.0, min(1.0, resp_rate))
    inter_rate = max(0.0, min(1.0, inter_rate))
    github_val = max(0.0, min(1.0, github_val))

    # --- A4: Additional engagement signals ---
    # Market demand: how many recruiters saved this candidate in last 30 days
    saved_30d = signals.get("saved_by_recruiters_30d", 0)
    market_demand = min(1.0, saved_30d / 10.0)   # 10+ saves → 1.0

    # Active seeker: how many roles they've applied to recently
    apps_30d = signals.get("applications_submitted_30d", 0)
    active_seeker = min(1.0, apps_30d / 5.0)     # 5+ applications → 1.0

    # Commitment: fraction of offers accepted (-1 = no prior offers → neutral)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    commit_score = float(offer_rate) if offer_rate >= 0 else 0.5
    commit_score = max(0.0, min(1.0, commit_score))

    # Response speed: how quickly they reply to recruiters
    avg_hours = signals.get("avg_response_time_hours", 24)
    if avg_hours <= 4:
        speed_score = 1.0
    elif avg_hours <= 24:
        speed_score = 0.8
    elif avg_hours <= 72:
        speed_score = 0.5
    else:
        speed_score = 0.3

    # Profile completeness: how much effort put into their profile
    completeness = signals.get("profile_completeness_score", 50) / 100.0
    completeness = max(0.0, min(1.0, completeness))

    # Weighted engagement — original signals get 60%, new A4 signals get 40%
    engagement_score = (
        0.25 * resp_rate +
        0.20 * inter_rate +
        0.15 * github_val +
        0.15 * market_demand +
        0.10 * commit_score +
        0.10 * speed_score +
        0.05 * active_seeker
        # completeness used implicitly via market_demand and resp_rate
    )
    # Slight boost for very complete profiles
    completeness_boost = 0.02 * completeness

    return min(1.0, (location_score + activity_score + engagement_score) / 3.0 + completeness_boost)


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
# 8. Assessment Score (5%) — A3: NEW — previously unused verified signal
# ---------------------------------------------------------------------------
def calculate_assessment_score(signals):
    """
    Uses Redrob's verified per-skill assessment scores (0–100 each).
    Only counts assessments for JD-relevant skills.
    Returns 0.5 (neutral) if no assessments taken.

    This is the ONLY verified signal in the dataset — self-reported skills
    can be fabricated; assessment scores cannot.
    """
    assessments = signals.get("skill_assessment_scores", {})
    if not assessments:
        return 0.5  # neutral — candidate hasn't taken any assessments

    # Find assessments that match JD-relevant skill areas
    relevant_scores = []
    for skill_name, score in assessments.items():
        skill_lower = skill_name.lower()
        if any(jd_kw in skill_lower for jd_kw in JD_ASSESSMENT_KEYWORDS):
            relevant_scores.append(float(score))

    if not relevant_scores:
        return 0.5  # has assessments but none in JD-relevant areas

    avg_score = sum(relevant_scores) / len(relevant_scores)
    return avg_score / 100.0  # normalize to [0.0, 1.0]


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
def calculate_confidence(scores_dict, final_score):
    """
    Computes a calibrated confidence percentage based on final score band
    and inter-signal agreement:
      Strong candidates (final >= 0.70): 85–97%
      Good candidates  (final >= 0.65): 72–85%
      Moderate         (final >= 0.60): 62–72%
      Weak             (final <  0.60): 55–62%
    Agreement (low std-dev across signals) positions the score within the band.
    """
    values = list(scores_dict.values())
    if not values:
        return 75
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_dev = math.sqrt(variance)
    agreement = max(0.0, 1.0 - std_dev * 4.0)

    if final_score >= 0.70:
        low, high = 85, 97
    elif final_score >= 0.65:
        low, high = 72, 85
    elif final_score >= 0.60:
        low, high = 62, 72
    else:
        low, high = 55, 62

    confidence = low + agreement * (high - low)
    return int(min(99, max(55, confidence)))


# ---------------------------------------------------------------------------
# 11. Evidence-Based Reasoning Generator — A5: rank-aware, varied structure
# ---------------------------------------------------------------------------
def _extract_tech_names(combined_text):
    """Scan career text for specific named technologies and return display names."""
    found = []
    for kw, display in TECH_DISPLAY.items():
        if kw in combined_text and display not in found:
            found.append(display)
    return found[:3]


def _detect_scale(combined_text):
    """Return True if career text contains evidence of working at scale."""
    scale_signals = [
        "million", "billion", "at scale", "high throughput",
        "large scale", "massive scale", "real-time"
    ]
    return any(sig in combined_text for sig in scale_signals)


def _build_concerns(signals, s_experience, years_exp, activity_score):
    """Build a list of honest concern strings for a candidate."""
    concerns = []
    notice = signals.get("notice_period_days", 60)
    resp_rate = float(signals.get("recruiter_response_rate", 0.5))

    if notice > 90:
        concerns.append(f"very long notice period ({notice} days)")
    elif notice > 60:
        concerns.append(f"notice period of {notice} days")

    if activity_score < 0.4:
        concerns.append("low recent platform activity")

    if years_exp < 4.0:
        concerns.append(f"limited experience ({years_exp:.1f} yrs vs 5–9 yr target)")
    elif years_exp > 13.0:
        concerns.append(f"significantly over-experienced ({years_exp:.1f} yrs)")

    if resp_rate < 0.3:
        concerns.append(f"low recruiter response rate ({resp_rate:.0%})")

    return concerns


def generate_reasoning(candidate, semantic_score, skills_score, career_relevance_score,
                        production_score, assessment_score, final_score, rank=None):
    """
    A5: Generates a rank-aware, candidate-specific, evidence-driven explanation.
    - Rank 1–15: Strong positive opener, lead with evidence and tech stack
    - Rank 16–50: Balanced opener, evidence + one flag if present
    - Rank 51–100: Explicitly surfaces concerns, honest tone
    - No fixed template: sentence structure varies across score bands
    """
    prof = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    years_exp = prof.get("years_of_experience", 0.0)
    current_title = prof.get("current_title", "Engineer")
    notice = signals.get("notice_period_days", 60)
    resp_rate = float(signals.get("recruiter_response_rate", 0.5))
    github_score = signals.get("github_activity_score", -1)
    open_to_work = signals.get("open_to_work_flag", False)

    last_act = parse_date_local(signals.get("last_active_date"))
    if last_act:
        days_inactive = (datetime(2026, 6, 17) - last_act).days
        activity_score = 1.0 if days_inactive <= 30 else (0.8 if days_inactive <= 90 else (0.5 if days_inactive <= 180 else 0.1))
    else:
        activity_score = 0.5

    combined_career = " ".join(
        role.get("description", "") for role in career
    ).lower()

    # Find evidenced capabilities
    caps_evidenced = [
        cap_name for cap_name, cap_keywords in CAPABILITIES.items()
        if any(kw in combined_career for kw in cap_keywords)
    ]

    tech_names = _extract_tech_names(combined_career)
    at_scale = _detect_scale(combined_career)

    # Build action clause
    if caps_evidenced:
        action_phrases = [
            CAP_ACTIONS[cap][0] for cap in caps_evidenced[:2] if cap in CAP_ACTIONS
        ]
        if len(action_phrases) >= 2:
            work_clause = f"built {action_phrases[0]} and {action_phrases[1]}"
        elif len(action_phrases) == 1:
            work_clause = f"specialized in {action_phrases[0]}"
        else:
            work_clause = "applied ML and data engineering practitioner"
    else:
        work_clause = "applied ML and data engineering practitioner"

    tech_clause = f" using {', '.join(tech_names[:2])}" if tech_names else ""

    prod_clause = ""
    if at_scale and production_score >= 0.5:
        prod_clause = " serving production workloads at scale"
    elif at_scale:
        prod_clause = " with demonstrated scale experience"
    elif production_score >= 0.5:
        prod_clause = " with strong production deployment experience"
    elif production_score >= 0.25:
        prod_clause = " with production deployment exposure"

    # Calibrated confidence
    confidence = calculate_confidence(
        {
            "semantic": semantic_score,
            "skills": skills_score,
            "career_relevance": career_relevance_score,
            "production": production_score,
        },
        final_score,
    )

    concerns = _build_concerns(signals, None, years_exp, activity_score)

    # --- A5: Rank-aware varied reasoning structure ---
    if rank is None or rank <= 15:
        # Top tier: lead with strength, concise
        availability = "immediately available" if notice <= 30 else f"{notice}-day notice"
        github_note = f"; active open-source contributor (GitHub: {github_score}/100)" if github_score >= 60 else ""
        reasoning = (
            f"{years_exp:.1f}-yr {current_title} who {work_clause}{tech_clause}{prod_clause}. "
            f"{availability}{github_note}. "
            f"(Confidence: {confidence}%)"
        )

    elif rank <= 50:
        # Mid tier: balanced — strengths then any single concern
        if resp_rate > 0.6:
            engage_note = "high recruiter engagement"
        elif github_score >= 50:
            engage_note = "strong open-source presence"
        elif open_to_work:
            engage_note = "actively marked open-to-work"
        else:
            engage_note = "moderate platform engagement"

        concern_note = ""
        if concerns:
            concern_note = f" Note: {concerns[0]}."

        reasoning = (
            f"Has {work_clause}{tech_clause}{prod_clause}, "
            f"with {years_exp:.1f} years as a {current_title} and {engage_note}.{concern_note} "
            f"(Confidence: {confidence}%)"
        )

    else:
        # Lower tier: honest — surface all significant concerns
        if concerns:
            concern_str = "; ".join(concerns)
            reasoning = (
                f"Adjacent profile: {work_clause}{tech_clause}. "
                f"Ranked lower due to: {concern_str}. "
                f"{years_exp:.1f} yrs experience as {current_title}. "
                f"(Confidence: {confidence}%)"
            )
        else:
            reasoning = (
                f"Borderline fit: {work_clause}{tech_clause}{prod_clause}. "
                f"{years_exp:.1f} yrs as {current_title}, but limited signal strength "
                f"on key JD requirements. "
                f"(Confidence: {confidence}%)"
            )

    return reasoning


# ---------------------------------------------------------------------------
# 12. Master evaluate_candidate — Full Scoring
# ---------------------------------------------------------------------------
def evaluate_candidate(candidate, semantic_score, deep=False, rank=None):
    """
    Evaluates a candidate profile against all scoring criteria.

    Args:
        candidate: The full candidate JSON object.
        semantic_score: Pre-computed cosine similarity score (float).
        deep: If True, applies career relevance, production, and assessment
              scoring (Stage 2 only). If False, uses fast approximations (Stage 1).
        rank: Optional rank position (used for rank-aware reasoning in Stage 2).

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

    # Schema-aware behavioral scoring (A4 expanded)
    behavior_dict = dict(signals)
    behavior_dict["location"] = prof.get("location", "")
    behavior_dict["country"] = prof.get("country", "")
    s_behavior = calculate_behavioral_score(behavior_dict)
    s_availability = calculate_availability_score(signals)

    # Deep Stage 2 scores (capability-based, expensive text scanning)
    if deep:
        career_relevance = calculate_career_relevance_score(career)
        s_production = calculate_production_score(career)
        s_assessment = calculate_assessment_score(signals)  # A3: new verified signal

        # Blend cosine similarity with capability relevance inside semantic score
        blended_semantic = 0.70 * semantic_score + 0.30 * career_relevance

        # Keyword stuffing detection (density-based)
        is_stuffing = detect_keyword_stuffing(skills, semantic_score, career)

        # 7-component formula (A3: assessment replaces hidden talent)
        raw_score = (
            0.25 * blended_semantic +
            0.20 * s_skills +
            0.15 * s_experience +
            0.15 * s_behavior +
            0.10 * s_availability +
            0.10 * s_production +
            0.05 * s_assessment
        )

        final_score = raw_score * 0.5 if is_stuffing else raw_score

        reasoning = generate_reasoning(
            candidate, blended_semantic, s_skills, career_relevance,
            s_production, s_assessment, final_score, rank=rank
        )
    else:
        # Stage 1 fast approximation (no text scanning)
        career_relevance = 0.0
        s_production = 0.0
        s_assessment = 0.0
        is_stuffing = False

        # Simplified fast score
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
