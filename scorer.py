# scorer.py
"""
Scoring module for the Redrob candidate discovery ranking engine.
Implements the revised 6-component hierarchical scoring formula,
keyword-stuffing detection, and custom reasoning generator.
"""

from datetime import datetime
import dateutil.parser
from skills_config import (
    CORE_CONCEPTS, AI_TITLE_KEYWORDS, GENERIC_TECH_TITLES, 
    IRRELEVANT_TITLES, TIER5_CAREER_KEYWORDS
)
from honeypot_detector import parse_date

# Let's write a simple helper to parse dates locally to keep scorer self-contained
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

def calculate_trust_factor(duration_months, endorsements):
    """Calculates skill trust multiplier to discount keyword stuffers."""
    # Min 1.0 at 12 months, scaling down for shorter durations
    duration_factor = min(1.0, duration_months / 12.0)
    # Scale from 0.5 to 1.0 based on endorsements (min 1.0 at 10 endorsements)
    endorsement_factor = 0.5 + 0.5 * min(1.0, endorsements / 10.0)
    return duration_factor * endorsement_factor

def calculate_skills_score(skills):
    """
    Computes a concept-based semantic skill score.
    Groups candidate skills into JD core requirements and weights by duration/endorsements.
    """
    category_scores = {}
    
    # Initialize scores for all concept categories
    for cat in CORE_CONCEPTS:
        category_scores[cat] = 0.0
        
    for skill in skills:
        name = skill.get("name", "").lower()
        proficiency_str = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)
        
        # Determine proficiency multiplier
        prof_mult = 0.2
        if proficiency_str == "expert":
            prof_mult = 1.0
        elif proficiency_str == "advanced":
            prof_mult = 0.8
        elif proficiency_str == "intermediate":
            prof_mult = 0.5
            
        trust = calculate_trust_factor(duration, endorsements)
        skill_val = prof_mult * trust
        
        # Match against CORE_CONCEPTS keywords
        for cat, config in CORE_CONCEPTS.items():
            if name in config["keywords"] or any(kw in name for kw in config["keywords"]):
                # Accumulate score for this category (capped at 1.0)
                category_scores[cat] = min(1.0, category_scores[cat] + skill_val)
                
    # Compute weighted average of category scores
    total_score = 0.0
    total_weight = 0.0
    for cat, config in CORE_CONCEPTS.items():
        w = config["weight"]
        total_score += category_scores[cat] * w
        total_weight += w
        
    return total_score / total_weight if total_weight > 0 else 0.0

def calculate_title_score(profile, career_history):
    """
    Calculates title alignment score (15% weight).
    Verifies tech vs non-tech title, maps generic titles, and detects Tier-5 upgrades.
    """
    curr_title = profile.get("current_title", "").lower()
    
    # Check current title
    if any(kw in curr_title for kw in AI_TITLE_KEYWORDS):
        base_score = 1.0
    elif any(kw in curr_title for kw in GENERIC_TECH_TITLES):
        base_score = 0.7
    elif any(kw in curr_title for kw in IRRELEVANT_TITLES):
        base_score = 0.1
    else:
        # Check if generic software/tech role
        base_score = 0.5

    # Tier-5 Upgrade: Check if they are a Software/Generic engineer who actually built matching systems
    has_tier5_experience = False
    tier5_details = []
    
    for role in career_history:
        desc = role.get("description", "").lower()
        role_title = role.get("title", "").lower()
        
        # Check if generic title in career history
        if any(kw in role_title for kw in GENERIC_TECH_TITLES):
            for kw in TIER5_CAREER_KEYWORDS:
                if kw in desc:
                    has_tier5_experience = True
                    tier5_details.append(kw)
                    
    if has_tier5_experience:
        # Boost generic software engineer title to recognize hidden matching system experience
        base_score = min(1.0, base_score + 0.25)
        
    return base_score

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

def calculate_behavioral_score(signals):
    """
    Combines platform activity, recruiter engagement, and location alignment.
    Ensures location acts only as a minor optimizer (5% overall score).
    """
    # 1. Location Score (5% of overall score, represents 1/3 of behavioral signal)
    loc = signals.get("location", "").lower()
    country = signals.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)
    
    # Standardize locations
    is_preferred = any(city in loc for city in ["noida", "pune", "ncr", "delhi ncr", "gurgaon", "ghaziabad", "faridabad"])
    is_tier1 = any(city in loc for city in ["mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata"])
    
    if is_preferred:
        location_score = 1.0
    elif is_tier1:
        location_score = 0.9
    else:
        location_score = 0.9 if relocate else 0.8
        
    # 2. Activity Score
    last_act = parse_date_local(signals.get("last_active_date"))
    if last_act:
        # Current date in simulation is June 16, 2026
        days_inactive = (datetime(2026, 6, 16) - last_act).days
        if days_inactive <= 30:
            activity_score = 1.0
        elif days_inactive <= 90:
            activity_score = 0.8
        elif days_inactive <= 180:
            activity_score = 0.5
        else:
            activity_score = 0.1
    else:
        activity_score = 0.5 # Neutral if missing
        
    # 3. Engagement Score
    resp_rate = signals.get("recruiter_response_rate", 0.5)
    inter_rate = signals.get("interview_completion_rate", 0.5)
    github_score = signals.get("github_activity_score", -1)
    
    # Map github activity
    if github_score >= 0:
        github_val = github_score / 100.0
    else:
        github_val = 0.5 # Neutral
        
    engagement_score = (resp_rate + inter_rate + github_val) / 3.0
    
    # Combine (each 1/3 weight in behavioral score)
    return (location_score + activity_score + engagement_score) / 3.0

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
        
    # Boost if open to work, downweight slightly if not
    multiplier = 1.1 if open_to_work else 0.9
    return min(1.0, max(0.0, base * multiplier))

def detect_keyword_stuffing(skills, semantic_score):
    """Checks for candidates with massive skill lists but low semantic relevance in description."""
    # Count skills with expert/advanced proficiency
    stuffed_skills_count = sum(
        1 for s in skills 
        if s.get("proficiency", "").lower() in ["expert", "advanced"]
    )
    
    # If they list a high number of skills (e.g. >= 10) but their Career Semantic Score is low (< 0.42)
    if stuffed_skills_count >= 10 and semantic_score < 0.42:
        return True
    return False

def generate_reasoning(candidate, semantic_score, skills_score, final_score):
    """Generates a structured recruiter justification detailing strengths, experience, behavior, and risks."""
    prof = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    
    years_exp = prof.get("years_of_experience", 0.0)
    current_title = prof.get("current_title", "Engineer")
    notice = signals.get("notice_period_days", 60)
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    
    # Identify key strengths based on matched categories
    strengths = []
    skill_names = [s.get("name", "").lower() for s in skills]
    
    # Check specific matching components
    if any(kw in s for s in skill_names for kw in CORE_CONCEPTS["retrieval_rag"]["keywords"]):
        strengths.append("retrieval systems")
    if any(kw in s for s in skill_names for kw in CORE_CONCEPTS["vector_databases"]["keywords"]):
        strengths.append("vector databases")
    if any(kw in s for s in skill_names for kw in CORE_CONCEPTS["ranking_evaluation"]["keywords"]):
        strengths.append("ranking systems")
    if any(kw in s for s in skill_names for kw in CORE_CONCEPTS["llm_dl"]["keywords"]):
        strengths.append("LLM fine-tuning")
        
    # Fallback if list is empty
    if not strengths:
        strengths = ["applied machine learning", "data engineering"]
        
    strengths_str = ", ".join(strengths[:3])
    
    # Availability statement
    availability_str = "high active presence" if resp_rate > 0.6 else "good platform engagement"
    
    # Risk/concern statement
    risk_str = ""
    if notice > 60:
        risk_str = f"A long notice period of {notice} days is noted as a concern."
    elif notice == 60:
        risk_str = "A 60-day notice period is noted as a minor concern."
        
    # Assemble sentence
    reasoning = (
        f"Candidate ranked highly due to strong experience in {strengths_str}, "
        f"{years_exp:.1f} years of relevant engineering experience as a {current_title}, "
        f"and {availability_str}."
    )
    if risk_str:
        reasoning += f" {risk_str}"
        
    return reasoning

def evaluate_candidate(candidate, semantic_score):
    """
    Evaluates a candidate profile against all scoring criteria.
    Returns:
        final_score (float): The combined score.
        reasoning (str): Text explanation of the score.
        is_stuffing (bool): If keyword stuffing was detected.
    """
    prof = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    
    # Calculate each score component
    s_skills = calculate_skills_score(skills)
    s_title = calculate_title_score(prof, career)
    s_experience = calculate_experience_score(prof.get("years_of_experience", 0.0))
    
    # Pass profile location/country/relocate fields into redrob_signals dictionary for behavioral scorer
    behavior_dict = dict(signals)
    behavior_dict["location"] = prof.get("location", "")
    behavior_dict["country"] = prof.get("country", "")
    
    s_behavior = calculate_behavioral_score(behavior_dict)
    s_availability = calculate_availability_score(signals)
    
    # Weighted score calculation
    raw_score = (
        0.25 * semantic_score +
        0.20 * s_skills +
        0.15 * s_title +
        0.15 * s_experience +
        0.15 * s_behavior +
        0.10 * s_availability
    )
    
    # Keyword Stuffing Detection
    is_stuffing = detect_keyword_stuffing(skills, semantic_score)
    final_score = raw_score
    if is_stuffing:
        final_score *= 0.5
        
    # Generate justification
    reasoning = generate_reasoning(candidate, semantic_score, s_skills, final_score)
    
    return final_score, reasoning, is_stuffing
