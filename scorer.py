# scorer.py
"""
Candidate Intelligence Engine - Scoring Module (Phase 3)

9-component hierarchical scoring formula:
  25% Career Semantic Score      (cosine sim + semantic capability relevance)
  18% Concept Skill Score        (skill category overlap, extended trust factor)
  13% Behavioral Score           (engagement, location, activity — A4 expanded)
  12% Experience Score           (Gaussian decay centred at 7 years)
  10% Production Readiness Score (12 independent dimensions)
   7% Availability Score         (notice period + open-to-work)
   5% Assessment Score           (verified platform scores — A3)
   5% Career Progression Score   (title level trajectory, tenure stability)
   5% Learning Velocity Score    (technology generation evolution)

Quality improvements:
  - ALL randomness removed: same candidate always produces same score + reasoning
  - Gaussian experience decay replaces piecewise hard thresholds
  - Logistic keyword-stuffing penalty replaces binary 0.5 multiplier
  - Extended trust factor uses career evidence, not just duration + endorsements
  - Semantic capability score accepted from rank.py (dot-product, < 5ms)
  - Ownership verb detection rewards high-evidence profiles
  - Career consistency scoring penalises domain drift
  - Structured Strengths / Concerns / Assessment reasoning format
"""

import math
from datetime import datetime
import dateutil.parser

from skills_config import (
    CORE_CONCEPTS, AI_TITLE_KEYWORDS, GENERIC_TECH_TITLES,
    IRRELEVANT_TITLES, TIER5_CAREER_KEYWORDS,
    CAPABILITIES, PRODUCTION_KEYWORDS, TECH_DISPLAY, CAP_ACTIONS,
    CAPABILITY_DESCRIPTIONS, TITLE_LEVEL_MAP, TECH_GENERATIONS,
    OWNERSHIP_VERBS, OWNERSHIP_VERBS_HIGH, OWNERSHIP_VERBS_MED, OWNERSHIP_VERBS_LOW,
    DOMAIN_GROUPS, PROJECT_COMPLEXITY_KEYWORDS, JD_ALIGNED_TECHS,
)

# ---------------------------------------------------------------------------
# JD-relevant assessment skill keywords
# ---------------------------------------------------------------------------
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
# 1. Career Semantic Score helper — capability keyword fallback
# ---------------------------------------------------------------------------
def calculate_career_relevance_score(career_history):
    """
    Keyword-based capability detection (fallback when semantic scores unavailable).
    Used in Stage 1 fast mode. Returns float in [0, 1].
    """
    combined = " ".join(
        role.get("description", "") for role in career_history
    ).lower()

    capabilities_found = sum(
        1 for cap_keywords in CAPABILITIES.values()
        if any(kw in combined for kw in cap_keywords)
    )
    return capabilities_found / len(CAPABILITIES) if CAPABILITIES else 0.0


# ---------------------------------------------------------------------------
# 2. Evidence Strength — Tiered ownership verb scoring (Phase 4)
# ---------------------------------------------------------------------------
def calculate_evidence_strength(career_history):
    """
    Tiered ownership verb detection. Distinguishes strong authorship language
    ("architected", "built from scratch") from generic execution language
    ("built", "deployed") and passive filler ("worked on", "helped with").

    Score = (high_hits * 1.0 + med_hits * 0.5 - low_hits * 0.15) / 5.0
    Clamped to [0, 1]. 5+ high-value claims = 1.0.
    """
    if not career_history:
        return 0.0
    combined = " ".join(
        role.get("description", "").lower() for role in career_history
    )
    high_hits = sum(1 for v in OWNERSHIP_VERBS_HIGH if v in combined)
    med_hits  = sum(1 for v in OWNERSHIP_VERBS_MED  if v in combined)
    low_hits  = sum(1 for v in OWNERSHIP_VERBS_LOW  if v in combined)

    raw = high_hits * 1.0 + med_hits * 0.5 - low_hits * 0.15
    return max(0.0, min(1.0, raw / 5.0))


# ---------------------------------------------------------------------------
# 3. Production Readiness Score (12%) — density scoring per dimension
# ---------------------------------------------------------------------------
def calculate_production_score(career_history):
    """
    Evaluates production deployment maturity across 12 conceptual dimensions.
    Each dimension now uses DENSITY scoring instead of binary present/absent:
      0 hits → 0.0
      1 hit  → 0.33
      2 hits → 0.67
      3+ hits → 1.0
    This rewards descriptions with multiple corroborating production signals
    over single-keyword mentions. Score = mean across all 12 dimensions.
    """
    combined = " ".join(
        role.get("description", "") for role in career_history
    ).lower()

    dim_scores = [
        min(1.0, sum(1 for kw in dim_kws if kw in combined) / 3.0)
        for dim_kws in PRODUCTION_KEYWORDS.values()
    ]
    return sum(dim_scores) / len(dim_scores) if dim_scores else 0.0


# ---------------------------------------------------------------------------
# 4. Skill Trust Factor — extended with career evidence
# ---------------------------------------------------------------------------
def calculate_trust_factor(skill_name, duration_months, endorsements,
                            career_history=None, current_title=""):
    """
    Extended trust multiplier for a skill. Goes beyond duration + endorsements
    to check whether the skill actually appears in career descriptions.

    Evidence tiers:
      - Skill in career + title alignment → 1.3x boost (strong evidence)
      - Skill only in career descriptions → 1.15x boost (moderate evidence)
      - Skill has duration > 0 but no career evidence → 1.0x (baseline)
      - Skill only in skills section, duration = 0 → 0.6x (weak evidence)
    """
    duration_factor = min(1.0, duration_months / 12.0)
    endorsement_factor = 0.5 + 0.5 * min(1.0, endorsements / 10.0)
    base_trust = duration_factor * endorsement_factor

    if not career_history:
        return base_trust

    combined_career = " ".join(
        (role.get("description", "") + " " + role.get("title", "")).lower()
        for role in career_history
    )
    skill_lower = skill_name.lower()
    career_evidence = skill_lower in combined_career
    title_alignment = skill_lower in current_title.lower()

    if career_evidence and title_alignment:
        multiplier = 1.3
    elif career_evidence:
        multiplier = 1.15
    elif duration_months > 0:
        multiplier = 1.0
    else:
        multiplier = 0.6  # declared only, zero evidence

    return min(1.0, base_trust * multiplier)


# ---------------------------------------------------------------------------
# 5. Concept Skill Score (18%)
# ---------------------------------------------------------------------------
def calculate_skills_score(skills, career_history=None, current_title=""):
    """
    Concept-based semantic skill score. Groups candidate skills into JD core
    categories, weighted by the extended trust factor.
    """
    category_scores = {cat: 0.0 for cat in CORE_CONCEPTS}

    for skill in skills:
        name = skill.get("name", "")
        proficiency_str = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        prof_mult = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5}.get(
            proficiency_str, 0.2
        )
        trust = calculate_trust_factor(
            name, duration, endorsements,
            career_history=career_history,
            current_title=current_title
        )
        skill_val = prof_mult * trust

        name_lower = name.lower()
        for cat, config in CORE_CONCEPTS.items():
            if name_lower in config["keywords"] or any(
                kw in name_lower for kw in config["keywords"]
            ):
                category_scores[cat] = min(1.0, category_scores[cat] + skill_val)

    total_score = sum(
        category_scores[cat] * CORE_CONCEPTS[cat]["weight"]
        for cat in CORE_CONCEPTS
    )
    total_weight = sum(c["weight"] for c in CORE_CONCEPTS.values())
    return total_score / total_weight if total_weight > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. Experience Score (12%) — Gaussian decay centred at 7 years
# ---------------------------------------------------------------------------
def calculate_experience_score(years_exp):
    """
    Smooth Gaussian decay instead of hard piecewise thresholds.
    Centred at 7.0 years (midpoint of the 5-9 yr JD target).
    σ = 3.5 → a 12-year veteran scores ~0.61, a 15-year veteran ~0.30.
    No arbitrary cliff edges.
    """
    return math.exp(-((years_exp - 7.0) ** 2) / (2.0 * (3.5 ** 2)))


# ---------------------------------------------------------------------------
# 7. Career Progression Score (5%)
# ---------------------------------------------------------------------------
def _get_title_level(title):
    """Map a job title to a seniority level integer (1–6)."""
    title_lower = title.lower()
    best_level = 2  # default IC
    for keyword, level in TITLE_LEVEL_MAP.items():
        if keyword in title_lower:
            best_level = max(best_level, level)
    return best_level


def _get_role_start_year(role):
    try:
        return int(str(role.get("start_date", "2000"))[:4])
    except Exception:
        return 2000


def _calculate_role_impact(role):
    """
    Scores a single role's organisational impact using description richness
    and scale-of-work indicators. Used as a company-progression proxy.
    """
    desc = role.get("description", "").lower()
    word_count = len(desc.split())
    richness = min(1.0, word_count / 120.0)  # 120+ words = max richness

    impact_words = [
        "billion", "million", "global", "enterprise", "large-scale",
        "distributed", "platform", "infrastructure", "at scale",
    ]
    impact_hits = sum(1 for w in impact_words if w in desc)
    impact_score = min(1.0, impact_hits / 3.0)  # 3+ = full score

    return 0.55 * richness + 0.45 * impact_score


def calculate_progression_score(career_history):
    """
    Rewards genuine upward career movement.
    Signals: net title level gain, tenure stability, peak seniority,
    no large drops, and role-impact progression (company intelligence proxy).
    Returns float in [0, 1].
    """
    if not career_history:
        return 0.3

    sorted_roles = sorted(career_history, key=_get_role_start_year)
    levels = [_get_title_level(r.get("title", "")) for r in sorted_roles]

    if len(levels) == 1:
        return min(1.0, levels[0] / 5.0)

    # Net progression: how many seniority levels gained overall?
    net_gain = levels[-1] - levels[0]
    progression_score = min(1.0, max(0.0, (net_gain + 2) / 6.0))

    # Tenure stability: average role duration (target ≥ 24 months)
    durations = [r.get("duration_months", 12) for r in sorted_roles]
    avg_duration = sum(durations) / len(durations)
    stability_score = min(1.0, avg_duration / 24.0)

    # Consistency: penalise large backward steps (more than 1 level drop)
    drops = sum(
        1 for i in range(1, len(levels))
        if levels[i] < levels[i - 1] - 1
    )
    consistency_score = max(0.0, 1.0 - drops * 0.25)

    # Peak seniority bonus
    peak_score = min(1.0, max(levels) / 5.0)

    # Role-impact progression: did responsibilities grow across roles?
    role_impacts = [_calculate_role_impact(r) for r in sorted_roles]
    mid = max(1, len(role_impacts) // 2)
    early_impact = sum(role_impacts[:mid]) / mid
    late_impact = sum(role_impacts[mid:]) / max(1, len(role_impacts) - mid)
    impact_progression = min(1.0, max(0.0, (late_impact - early_impact + 0.5) / 1.0))

    return (
        0.35 * progression_score +
        0.20 * stability_score +
        0.18 * consistency_score +
        0.15 * peak_score +
        0.12 * impact_progression   # company/role progression intelligence
    )


# ---------------------------------------------------------------------------
# 8. Learning Velocity Score (5%)
# ---------------------------------------------------------------------------
def calculate_learning_velocity(career_history, skills):
    """
    Measures how quickly a candidate adopts newer technologies.
    Phase 4: JD-aligned technologies (RAG, vLLM, QLoRA, vector DBs, etc.)
    receive a 1.5x multiplier in the generation count to reward candidates
    who are proficient in exactly the technologies the role requires.
    Returns float in [0, 1].
    """
    sorted_roles = sorted(career_history, key=_get_role_start_year)

    gen_by_period = []
    for role in sorted_roles:
        year = _get_role_start_year(role)
        desc = (role.get("description", "") + " " + role.get("title", "")).lower()
        role_max_gen = 1.0
        for tech, gen in TECH_GENERATIONS.items():
            if tech in desc:
                # JD-aligned techs get a 1.5x multiplier (capped at gen 5)
                effective_gen = gen * 1.5 if tech in JD_ALIGNED_TECHS else gen
                role_max_gen = max(role_max_gen, effective_gen)
        gen_by_period.append((year, role_max_gen))

    # Skills list enriches the picture (no dates, but reflects current knowledge)
    all_skill_text = " ".join(s.get("name", "").lower() for s in skills)
    max_skill_gen = 1.0
    for tech, gen in TECH_GENERATIONS.items():
        if tech in all_skill_text:
            effective_gen = gen * 1.5 if tech in JD_ALIGNED_TECHS else gen
            max_skill_gen = max(max_skill_gen, effective_gen)

    if not gen_by_period:
        return min(1.0, max_skill_gen / 7.5) * 0.6  # 7.5 = 5 * 1.5 max

    max_gen_reached = max(g for _, g in gen_by_period)
    max_gen_reached = max(max_gen_reached, max_skill_gen)

    # Peak generation score (0–7.5 effective range → 0–1)
    peak_score = min(1.0, max_gen_reached / 7.5)

    # Velocity: did they adopt newer tech over time?
    mid = max(1, len(gen_by_period) // 2)
    early_gens = [g for _, g in gen_by_period[:mid]]
    late_gens  = [g for _, g in gen_by_period[mid:]]
    if late_gens:
        early_avg = sum(early_gens) / len(early_gens)
        late_avg  = sum(late_gens) / len(late_gens)
        # Map delta (-7.5..+7.5) → (0..1)
        velocity_score = min(1.0, max(0.0, (late_avg - early_avg + 3.0) / 6.0))
    else:
        velocity_score = 0.5

    # Diversity: not stuck in a single era
    unique_gens = len(set(round(g) for _, g in gen_by_period))
    diversity_score = min(1.0, unique_gens / 3.0)

    return (
        0.40 * peak_score +
        0.40 * velocity_score +
        0.20 * diversity_score
    )


# ---------------------------------------------------------------------------
# 9. Career Consistency Score (internal modifier)
# ---------------------------------------------------------------------------
def calculate_consistency_score(career_history):
    """
    Rewards candidates whose career path stays within related technical domains.
    Penalises radical domain shifts into unrelated fields.
    Returns float in [0, 1].
    """
    if not career_history:
        return 0.5

    def get_domain(title):
        title_lower = title.lower()
        for domain, keywords in DOMAIN_GROUPS.items():
            if any(kw in title_lower for kw in keywords):
                return domain
        return "unknown"

    sorted_roles = sorted(career_history, key=_get_role_start_year)
    domains = [get_domain(r.get("title", "")) for r in sorted_roles]

    total = len(domains)
    if total == 0:
        return 0.5

    relevant_domains = {"ai_ml", "software_engineering", "data_engineering"}
    relevant_count = sum(1 for d in domains if d in relevant_domains)
    unrelated_count = sum(1 for d in domains if d == "unrelated")
    recent = domains[-min(3, total):]
    recent_relevance = sum(1 for d in recent if d in relevant_domains) / len(recent)

    return (
        0.50 * (relevant_count / total) +
        0.30 * recent_relevance +
        0.20 * max(0.0, 1.0 - (unrelated_count / total) * 2)
    )


# ---------------------------------------------------------------------------
# 10a. Project Complexity Score (2%) — Phase 4 addition
# ---------------------------------------------------------------------------
def calculate_project_complexity_score(career_history):
    """
    Measures engineering complexity beyond simple production keyword presence.
    5 dimensions each scored by hit density (0–1 per dim), averaged.
    Rewards: billion-scale work, distributed systems, production AI infra,
    high-throughput pipelines, and deep ownership language.
    Returns float in [0, 1].
    """
    if not career_history:
        return 0.0
    combined = " ".join(
        role.get("description", "") for role in career_history
    ).lower()

    dim_scores = [
        min(1.0, sum(1 for kw in dim_kws if kw in combined) / 2.0)
        for dim_kws in PROJECT_COMPLEXITY_KEYWORDS.values()
    ]
    return sum(dim_scores) / len(dim_scores) if dim_scores else 0.0


# ---------------------------------------------------------------------------
# 10. Behavioral Score (13%) — A4 expanded
# ---------------------------------------------------------------------------
def calculate_behavioral_score(signals):
    """
    Combines location, platform activity, and recruiter engagement.
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

    resp_rate = max(0.0, min(1.0, float(signals.get("recruiter_response_rate", 0.5))))
    inter_rate = max(0.0, min(1.0, float(signals.get("interview_completion_rate", 0.5))))
    github_score = signals.get("github_activity_score", -1)
    github_val = max(0.0, min(1.0, github_score / 100.0)) if github_score >= 0 else 0.5

    # A4 signals
    saved_30d = min(1.0, signals.get("saved_by_recruiters_30d", 0) / 10.0)
    apps_30d = min(1.0, signals.get("applications_submitted_30d", 0) / 5.0)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    commit_score = max(0.0, min(1.0, float(offer_rate))) if offer_rate >= 0 else 0.5
    avg_hours = signals.get("avg_response_time_hours", 24)
    if avg_hours <= 4:
        speed_score = 1.0
    elif avg_hours <= 24:
        speed_score = 0.8
    elif avg_hours <= 72:
        speed_score = 0.5
    else:
        speed_score = 0.3
    completeness = max(0.0, min(1.0, signals.get("profile_completeness_score", 50) / 100.0))

    engagement_score = (
        0.25 * resp_rate +
        0.20 * inter_rate +
        0.15 * github_val +
        0.15 * saved_30d +
        0.10 * commit_score +
        0.10 * speed_score +
        0.05 * apps_30d
    )
    completeness_boost = 0.02 * completeness
    return min(1.0, (location_score + activity_score + engagement_score) / 3.0 + completeness_boost)


# ---------------------------------------------------------------------------
# 11. Availability Score (7%)
# ---------------------------------------------------------------------------
def calculate_availability_score(signals):
    """Notice period suitability."""
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
# 12. Assessment Score (5%) — A3: verified platform scores
# ---------------------------------------------------------------------------
def calculate_assessment_score(signals):
    """
    Uses verified per-skill assessment scores (0–100 each).
    The only non-self-reported signal in the dataset.
    Returns 0.5 (neutral) if no assessments taken.
    """
    assessments = signals.get("skill_assessment_scores", {})
    if not assessments:
        return 0.5
    relevant_scores = [
        float(score) for skill_name, score in assessments.items()
        if any(jd_kw in skill_name.lower() for jd_kw in JD_ASSESSMENT_KEYWORDS)
    ]
    if not relevant_scores:
        return 0.5
    return sum(relevant_scores) / (len(relevant_scores) * 100.0)


# ---------------------------------------------------------------------------
# 13. Keyword Stuffing Detection — logistic penalty multiplier
# ---------------------------------------------------------------------------
def calculate_stuffing_multiplier(skills, semantic_score, career_history):
    """
    Detects keyword-heavy profiles with weak actual work evidence.
    Returns a CONTINUOUS penalty multiplier in [0.5, 1.0].
    Logistic decay: penalty ramps smoothly rather than a binary 50% cut.
    """
    total_skills = len(skills)
    if total_skills == 0:
        return 1.0  # no penalty

    high_prof_count = sum(
        1 for s in skills
        if s.get("proficiency", "").lower() in ["expert", "advanced"]
    )
    keyword_density = high_prof_count / total_skills

    combined = " ".join(
        role.get("description", "") for role in career_history
    ).lower()
    evidence_keywords = [
        "retrieval", "ranking", "recommendation", "vector", "embedding",
        "search", "nlp", "machine learning", "neural", "model"
    ]
    evidence_count = sum(1 for kw in evidence_keywords if kw in combined)
    has_strong_evidence = evidence_count >= 3

    # Stuffing signal: high density + low semantic + no career evidence
    if has_strong_evidence:
        stuffing_signal = 0.0
    else:
        stuffing_signal = keyword_density * max(0.0, 0.50 - semantic_score)

    # Logistic: multiplier ∈ [0.5, 1.0]
    #   stuffing_signal → 0   → multiplier ≈ 1.0 (no penalty)
    #   stuffing_signal → 0.1 → multiplier ≈ 0.75
    #   stuffing_signal → 0.2 → multiplier ≈ 0.50
    penalty = 0.5 / (1.0 + math.exp(-15.0 * (stuffing_signal - 0.10)))
    return max(0.5, min(1.0, 1.0 - penalty))


# ---------------------------------------------------------------------------
# 14. Confidence Score
# ---------------------------------------------------------------------------
def calculate_confidence(scores_dict, final_score):
    """
    Calibrated confidence % based on final score band and signal agreement.
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

    return int(min(99, max(55, low + agreement * (high - low))))


# ---------------------------------------------------------------------------
# 15. Reasoning helpers
# ---------------------------------------------------------------------------
def _extract_tech_names(combined_text):
    found = []
    for kw, display in TECH_DISPLAY.items():
        if kw in combined_text and display not in found:
            found.append(display)
    return found[:4]


def _detect_scale(combined_text):
    scale_signals = [
        "million", "billion", "at scale", "high throughput",
        "large scale", "massive scale", "real-time"
    ]
    return any(sig in combined_text for sig in scale_signals)


def _build_concerns(signals, years_exp, activity_score):
    concerns = []
    notice = signals.get("notice_period_days", 60)
    resp_rate = float(signals.get("recruiter_response_rate", 0.5))
    if notice > 90:
        concerns.append(f"very long notice period ({notice} days)")
    elif notice > 60:
        concerns.append(f"notice period of {notice} days")
    if activity_score < 0.4:
        concerns.append("low recent platform activity")
    if years_exp is not None:
        if years_exp < 4.0:
            concerns.append(f"limited experience ({years_exp:.1f} yrs vs 5–9 yr target)")
        elif years_exp > 13.0:
            concerns.append(f"significantly over-experienced ({years_exp:.1f} yrs)")
    if resp_rate < 0.3:
        concerns.append(f"low recruiter response rate ({resp_rate:.0%})")
    return concerns


# ---------------------------------------------------------------------------
# 16. Structured Reasoning Generator — fully deterministic, no randomness
# ---------------------------------------------------------------------------
def generate_reasoning(candidate, semantic_score, skills_score, career_relevance_score,
                        production_score, assessment_score, final_score, rank=None,
                        progression_score=0.0, velocity_score=0.0,
                        consistency_score=0.5, evidence_strength=0.0,
                        semantic_cap_scores=None, complexity_score=0.0):
    """
    Generates a structured, deterministic recruiter-style explanation.
    Format: Strengths | Concerns | Assessment
    No randomness — same candidate always produces the same explanation.
    """
    prof = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    years_exp = prof.get("years_of_experience", 0.0)
    current_title = prof.get("current_title", "Engineer")
    notice = signals.get("notice_period_days", 60)
    github_score = signals.get("github_activity_score", -1)
    open_to_work = signals.get("open_to_work_flag", False)

    last_act = parse_date_local(signals.get("last_active_date"))
    if last_act:
        days_inactive = (datetime(2026, 6, 17) - last_act).days
        activity_score = (1.0 if days_inactive <= 30
                          else 0.8 if days_inactive <= 90
                          else 0.5 if days_inactive <= 180
                          else 0.1)
    else:
        activity_score = 0.5

    combined_career = " ".join(
        role.get("description", "") for role in career
    ).lower()
    tech_names = _extract_tech_names(combined_career)
    at_scale = _detect_scale(combined_career)

    # --- Build Strengths list ---
    strengths = []

    # Semantic alignment
    if semantic_score >= 0.72:
        strengths.append("Strong semantic alignment with Senior AI Engineer JD")
    elif semantic_score >= 0.55:
        strengths.append("Solid semantic match with core JD requirements")

    # Capability evidence (from cap scores or keyword fallback)
    if semantic_cap_scores:
        # Pick the top-2 capabilities by semantic score
        top_caps = sorted(semantic_cap_scores.items(), key=lambda x: -x[1])[:2]
        cap_phrases = {
            "retrieval": "retrieval and semantic search",
            "ranking": "ranking and relevance scoring",
            "recommendation": "recommendation engine development",
            "matching": "entity and candidate matching",
            "vector_database": "vector database deployment",
            "llm_finetuning": "LLM fine-tuning and transformers",
        }
        for cap, score in top_caps:
            if score >= 0.30:
                strengths.append(cap_phrases.get(cap, cap).capitalize() + " experience")
    else:
        caps_evidenced = [
            cap for cap, cap_kws in CAPABILITIES.items()
            if any(kw in combined_career for kw in cap_kws)
        ]
        cap_labels = {
            "retrieval": "Retrieval and semantic search experience",
            "ranking": "Ranking and relevance scoring expertise",
            "recommendation": "Recommendation engine experience",
            "matching": "Candidate/entity matching infrastructure",
            "vector_database": "Vector database deployment experience",
            "llm_finetuning": "LLM fine-tuning and transformer expertise",
        }
        for cap in caps_evidenced[:2]:
            strengths.append(cap_labels.get(cap, cap))

    # Tech stack
    if tech_names:
        strengths.append(f"Tooling: {', '.join(tech_names[:3])}")

    # Production evidence
    if production_score >= 0.60:
        strengths.append(
            "Production deployment at scale" if at_scale
            else "Strong production engineering background"
        )
    elif production_score >= 0.30:
        strengths.append("Production deployment exposure")

    # Career signals
    if progression_score >= 0.70:
        strengths.append("Clear upward career trajectory")
    if velocity_score >= 0.70:
        strengths.append("Adopts cutting-edge technologies consistently")
    if github_score >= 60:
        strengths.append(f"Active open-source contributor (GitHub: {github_score}/100)")
    if assessment_score > 0.70:
        strengths.append(f"Strong verified assessment scores ({assessment_score:.0%})")
    if evidence_strength >= 0.60:
        strengths.append("High-ownership language in career descriptions")
    if complexity_score >= 0.50:
        strengths.append("Large-scale / distributed engineering evidence")
    elif complexity_score >= 0.30:
        strengths.append("Non-trivial engineering complexity evidenced")

    # --- Build Concerns list ---
    concerns = _build_concerns(signals, years_exp, activity_score)
    if consistency_score < 0.40:
        concerns.append("non-linear career path with domain shifts")
    if velocity_score < 0.30:
        concerns.append("limited adoption of modern AI/ML tooling")
    if production_score < 0.17:
        concerns.append("minimal production deployment evidence")

    # --- Overall Assessment ---
    if final_score >= 0.72:
        assessment = (
            f"Excellent fit — {years_exp:.1f}-yr {current_title} with "
            f"well-evidenced AI/ML experience and high JD alignment."
        )
    elif final_score >= 0.65:
        assessment = (
            f"Good candidate — {years_exp:.1f}-yr {current_title} with "
            f"relevant background; some areas could be stronger."
        )
    elif final_score >= 0.58:
        assessment = (
            f"Moderate fit — {years_exp:.1f}-yr {current_title} with "
            f"partial alignment; notable concerns present."
        )
    else:
        assessment = (
            f"Borderline candidate — {years_exp:.1f}-yr {current_title} "
            f"with limited alignment on key JD requirements."
        )

    # Availability
    if notice <= 30:
        avail = "immediately available"
    elif notice <= 60:
        avail = f"{notice}-day notice"
    else:
        avail = f"long notice period ({notice} days)"

    # Confidence
    confidence = calculate_confidence(
        {
            "semantic": semantic_score,
            "skills": skills_score,
            "career_relevance": career_relevance_score,
            "production": production_score,
        },
        final_score,
    )

    strengths_str = " | ".join(strengths[:5]) if strengths else "Moderate skill alignment"
    concerns_str = " | ".join(concerns[:3]) if concerns else "No major concerns"

    return (
        f"Strengths: {strengths_str}. "
        f"Concerns: {concerns_str}. "
        f"Assessment: {assessment} {avail.capitalize()}. "
        f"(Confidence: {confidence}%)"
    )


# ---------------------------------------------------------------------------
# 17. Master evaluate_candidate — Full Scoring
# ---------------------------------------------------------------------------
def evaluate_candidate(candidate, semantic_score, semantic_cap_scores=None,
                        deep=False, rank=None):
    """
    Evaluates a candidate against all scoring criteria.

    Args:
        candidate:          Full candidate JSON object.
        semantic_score:     Pre-computed cosine similarity (float).
        semantic_cap_scores: Dict {cap_name: dot_product_score} from rank.py
                             Stage 2 batch matrix multiply. None in Stage 1.
        deep:               If True, runs all 9 components (Stage 2 only).
                            If False, uses fast approximation (Stage 1).
        rank:               Optional rank position for reasoning tone.

    Returns:
        final_score (float), reasoning (str), stuffing_multiplier (float)
    """
    prof = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    current_title = prof.get("current_title", "")
    years_exp = prof.get("years_of_experience", 0.0)

    # --- Always computed (Stages 1 & 2) ---
    s_skills = calculate_skills_score(
        skills,
        career_history=career if deep else None,
        current_title=current_title
    )
    s_experience = calculate_experience_score(years_exp)
    behavior_dict = dict(signals)
    behavior_dict["location"] = prof.get("location", "")
    behavior_dict["country"] = prof.get("country", "")
    s_behavior = calculate_behavioral_score(behavior_dict)
    s_availability = calculate_availability_score(signals)

    if deep:
        # --- Deep Stage 2 scores ---
        s_production = calculate_production_score(career)
        s_assessment = calculate_assessment_score(signals)
        s_progression = calculate_progression_score(career)
        s_velocity = calculate_learning_velocity(career, skills)
        s_consistency = calculate_consistency_score(career)
        s_complexity = calculate_project_complexity_score(career)
        evidence_strength = calculate_evidence_strength(career)

        # Semantic capability relevance (primary if available, keyword fallback otherwise)
        if semantic_cap_scores is not None:
            # Weight top-2 capabilities more (semantic primary signal)
            sorted_caps = sorted(semantic_cap_scores.values(), reverse=True)
            if len(sorted_caps) >= 2:
                top2_avg = (sorted_caps[0] + sorted_caps[1]) / 2.0
                rest_avg = sum(sorted_caps[2:]) / max(1, len(sorted_caps) - 2)
                raw_cap = 0.60 * top2_avg + 0.40 * rest_avg
            else:
                raw_cap = sum(sorted_caps) / len(sorted_caps)
            career_relevance = min(1.0, raw_cap + 0.12 * evidence_strength)
        else:
            career_relevance = calculate_career_relevance_score(career)

        # Blended semantic: cosine + semantic capability relevance
        blended_semantic = 0.70 * semantic_score + 0.30 * career_relevance

        # Keyword stuffing — logistic multiplier
        stuffing_mult = calculate_stuffing_multiplier(skills, semantic_score, career)

        # 10-component formula (weights sum to 1.00)
        # Behavioral reduced 13% → 9%; weight redistributed to semantic, production, complexity
        raw_score = (
            0.27 * blended_semantic +
            0.17 * s_skills +
            0.12 * s_experience +
            0.12 * s_production +
            0.09 * s_behavior +
            0.06 * s_availability +
            0.05 * s_assessment +
            0.05 * s_progression +
            0.05 * s_velocity +
            0.02 * s_complexity
        )

        final_score = raw_score * stuffing_mult

        # Consistency acts as a gentle modifier (±3%)
        consistency_adj = (s_consistency - 0.5) * 0.06
        final_score = max(0.0, min(1.0, final_score + consistency_adj))

        reasoning = generate_reasoning(
            candidate,
            blended_semantic,
            s_skills,
            career_relevance,
            s_production,
            s_assessment,
            final_score,
            rank=rank,
            progression_score=s_progression,
            velocity_score=s_velocity,
            consistency_score=s_consistency,
            evidence_strength=evidence_strength,
            semantic_cap_scores=semantic_cap_scores,
            complexity_score=s_complexity,
        )
    else:
        # --- Stage 1 fast approximation (no text scanning) ---
        stuffing_mult = 1.0
        final_score = (
            0.35 * semantic_score +
            0.25 * s_skills +
            0.20 * s_experience +
            0.15 * s_behavior +
            0.05 * s_availability
        )
        reasoning = ""

    return final_score, reasoning, stuffing_mult
