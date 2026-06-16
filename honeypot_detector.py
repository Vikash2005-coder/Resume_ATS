# honeypot_detector.py
"""
Honeypot Detector for the Redrob candidate discovery ranking engine.
Identifies impossible/fake profiles and flags them for exclusion (Score = 0.0).
"""

from datetime import datetime
import dateutil.parser

def parse_date(date_str):
    """Parses date string to datetime object, returns None on failure."""
    if not date_str:
        return None
    try:
        return dateutil.parser.parse(date_str)
    except Exception:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None

def is_honeypot(candidate):
    """
    Evaluates candidate profile for impossible or fake properties.
    Returns:
        bool: True if candidate is a honeypot, False otherwise.
        list: List of strings detailing why the candidate was flagged.
    """
    reasons = []
    
    prof = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    
    years_of_experience = prof.get("years_of_experience", 0.0)
    
    # =========================================================================
    # Check 1: Expert Skill Duration Mismatch
    # =========================================================================
    # High-level technical skills listed as expert but with 0 duration
    expert_0_duration = []
    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        
        # Non-technical skills (languages, Photoshop) are excluded from strict technical check
        if name.lower() in ["english", "hindi", "bengali", "tamil", "telugu", "german", "french", "spanish"]:
            continue
            
        if proficiency == "expert" and duration == 0:
            expert_0_duration.append(name)
            
    # Flag if they have any expert technical skills with 0 months of experience
    if len(expert_0_duration) >= 1:
        reasons.append(f"Expert skills with 0 months experience: {expert_0_duration}")

    # =========================================================================
    # Check 2: Total Stated Experience vs Career History Duration Mismatch
    # =========================================================================
    total_months = 0
    for role in career:
        # Check if duration is listed
        total_months += role.get("duration_months", 0)
        
    total_years = total_months / 12.0
    # If the stated experience in profile differs from actual career history by more than 5.0 years
    if abs(years_of_experience - total_years) > 5.0:
        reasons.append(f"Stated experience ({years_of_experience} years) differs from career history total ({total_years:.2f} years)")

    # =========================================================================
    # Check 3: Date Timeline Anomalies
    # =========================================================================
    for role in career:
        title = role.get("title", "")
        company = role.get("company", "")
        start_date = parse_date(role.get("start_date"))
        end_date = parse_date(role.get("end_date"))
        duration = role.get("duration_months", 0)
        
        if start_date and end_date:
            if start_date > end_date:
                reasons.append(f"Role '{title}' at '{company}' start date ({role['start_date']}) is after end date ({role['end_date']})")
                
        # If start date is in the future relative to the current hackathon time (June 2026)
        if start_date and start_date.year > 2026:
            reasons.append(f"Role '{title}' at '{company}' start date ({role['start_date']}) is in the future")
            
        # Check if duration is negative
        if duration < 0:
            reasons.append(f"Role '{title}' at '{company}' has negative duration ({duration} months)")

    # =========================================================================
    # Check 4: Graduation Mismatch
    # =========================================================================
    # Look for full-time professional experience starting long before graduation
    grad_years = []
    for edu in education:
        deg = edu.get("degree", "").lower()
        # Look for bachelor or higher degrees
        if any(b in deg for b in ["b.e.", "b.tech", "b.sc", "b.a.", "b.com", "bachelor", "b.ba", "m.tech", "m.sc", "m.s.", "ph.d"]):
            end_yr = edu.get("end_year")
            if end_yr:
                grad_years.append(int(end_yr))
                
    if grad_years:
        earliest_grad_year = min(grad_years)
        for role in career:
            title = role.get("title", "").lower()
            start_dt = parse_date(role.get("start_date"))
            
            # Skip roles that look like internships, student work, or fellowships
            if any(i in title for i in ["intern", "trainee", "student", "part-time", "fellow", "research assistant"]):
                continue
                
            if start_dt and start_dt.year < earliest_grad_year - 4:
                reasons.append(f"Full-time role '{role['title']}' started in {start_dt.year} before graduation year {earliest_grad_year}")

    # =========================================================================
    # Check 5: Skill Stuffing
    # =========================================================================
    # A candidate with very little experience who claims expert level on many skills
    expert_skills = [s.get("name") for s in skills if s.get("proficiency", "").lower() == "expert"]
    if years_of_experience < 3.0 and len(expert_skills) >= 10:
        reasons.append(f"Stated experience is only {years_of_experience} years, but candidate claims 10+ expert skills: {expert_skills}")

    return len(reasons) > 0, reasons
