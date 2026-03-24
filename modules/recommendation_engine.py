"""
Module: Recommendation Engine
Layer 1 of the Attrition Mitigation Recommendation Engine.

Maps SHAP top risk-increasing factors to curated, evidence-based HR interventions.
Output is consumed directly by end_to_end_pipeline.py and the /api/recommend endpoint.
"""

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Priority thresholds (based on SHAP magnitude in model score space)
# ---------------------------------------------------------------------------
PRIORITY_CRITICAL = 0.25     # SHAP >= 0.25  → critical
PRIORITY_RECOMMENDED = 0.10  # SHAP >= 0.10  → recommended
# else                        → optional


# ---------------------------------------------------------------------------
# Intervention Lookup Table
# Feature name → {action, category, evidence}
# Priority is assigned dynamically based on SHAP magnitude at runtime.
# ---------------------------------------------------------------------------
INTERVENTION_TABLE: Dict[str, Dict[str, str]] = {

    # ── WORKLOAD DOMAIN ─────────────────────────────────────────────────────
    "Overtime_Hours": {
        "action": (
            "Cap weekly overtime to a maximum threshold and introduce flexible scheduling "
            "or compressed work-week policies to redistribute workload pressure."
        ),
        "category": "workload",
        "evidence": (
            "Excess overtime is a well-documented predictor of burnout and voluntary turnover "
            "(Pencavel, 2015; Bakker & Demerouti, 2007)."
        ),
    },
    "Overtime_Ratio": {
        "action": (
            "Review project scoping and team headcount to reduce the proportion of overtime "
            "relative to contracted hours. Consider hiring temporary contractors for peak periods."
        ),
        "category": "workload",
        "evidence": (
            "High overtime ratios indicate systemic understaffing rather than individual workload issues, "
            "requiring structural rather than individual-level intervention (Cooper & Lu, 2019)."
        ),
    },
    "Workload_Intensity": {
        "action": (
            "Conduct a workload audit across the team. Redistribute projects, defer non-urgent tasks, "
            "and set explicit boundaries on concurrent project assignments."
        ),
        "category": "workload",
        "evidence": (
            "Composite workload intensity is a robust predictor of intention-to-leave "
            "(Spector & Jex, 1998; Maslach et al., 2001)."
        ),
    },
    "Burnout_Risk": {
        "action": (
            "Initiate an immediate wellbeing check-in with the employee. Offer access to an Employee "
            "Assistance Programme (EAP), temporary workload reduction, and mandatory rest periods. "
            "Flag for manager coaching on workload distribution."
        ),
        "category": "workload",
        "evidence": (
            "Burnout is a critical precursor to voluntary resignation, with studies showing a "
            "3–5x higher turnover rate among burned-out employees (Leiter & Maslach, 2009)."
        ),
    },
    "Work_Life_Balance": {
        "action": (
            "Implement hybrid or remote work options. Enforce 'right to disconnect' policies outside "
            "contracted hours. Offer additional leave days or mental health days."
        ),
        "category": "workload",
        "evidence": (
            "Poor work-life balance is consistently ranked as one of the top three reasons for "
            "voluntary resignation in HR surveys (SHRM, 2023; Gallup, 2022)."
        ),
    },
    "Is_Overworked": {
        "action": (
            "Immediately reduce assigned work hours to within the 40-hour standard work week. "
            "Schedule a structured workload review with the line manager within two weeks."
        ),
        "category": "workload",
        "evidence": (
            "Employees consistently working over 45 hours per week show a 33% higher risk of "
            "attrition compared to those within normal hours (Pencavel, 2015)."
        ),
    },
    "Work_Hours_Per_Week": {
        "action": (
            "Review total contracted and actual hours. Reduce excess hours via task prioritisation "
            "frameworks (e.g., MoSCoW) or additional team resource allocation."
        ),
        "category": "workload",
        "evidence": (
            "Consistently high weekly hours degrade cognitive performance and decision-making quality, "
            "accelerating disengagement and eventual departure (Pencavel, 2015)."
        ),
    },
    "Projects_Handled": {
        "action": (
            "Reduce concurrent project assignments. Apply a WIP (Work-In-Progress) limit of no more "
            "than 3 simultaneous major projects per individual contributor."
        ),
        "category": "workload",
        "evidence": (
            "Context-switching across multiple concurrent projects reduces productivity by up to 40% "
            "and increases dissatisfaction (Meyer, 2019)."
        ),
    },

    # ── COMPENSATION DOMAIN ─────────────────────────────────────────────────
    "Monthly_Salary": {
        "action": (
            "Benchmark this employee's salary against current market rates (e.g., via Mercer or "
            "PayScale data). If below the 50th percentile for their role and seniority, initiate "
            "an off-cycle compensation review and adjustment."
        ),
        "category": "compensation",
        "evidence": (
            "Competitive compensation is the most cited factor in voluntary attrition, "
            "particularly in the first 3 years of tenure (SHRM Retention Survey, 2023)."
        ),
    },
    "Salary_Performance_Gap": {
        "action": (
            "Align pay with performance outcomes by introducing or recalibrating a performance-linked "
            "bonus structure. High performers below market pay are at the greatest flight risk."
        ),
        "category": "compensation",
        "evidence": (
            "A misalignment between perceived contribution and compensation is a primary driver of "
            "resentment and exit intention (Adams' Equity Theory, 1963; Milkovich et al., 2014)."
        ),
    },

    # ── CAREER GROWTH DOMAIN ─────────────────────────────────────────────────
    "Promotions": {
        "action": (
            "Review the employee's promotion history against their peer cohort. If overdue, initiate "
            "a formal career conversation and create a 90-day promotion readiness plan with clear, "
            "measurable milestones."
        ),
        "category": "career_growth",
        "evidence": (
            "Lack of career advancement is the second most cited reason for voluntary turnover "
            "in knowledge-worker industries (LinkedIn Workforce Report, 2023)."
        ),
    },
    "Training_Hours": {
        "action": (
            "Increase annual training allocation by a minimum of 20 hours. Enrol the employee in "
            "a structured learning pathway aligned to their career goals (e.g., technical certifications, "
            "leadership development programmes)."
        ),
        "category": "career_growth",
        "evidence": (
            "Organisations investing in employee L&D report 24% higher retention rates "
            "(LinkedIn Learning, 2023; Noe, 2020)."
        ),
    },
    "Training_Per_Year": {
        "action": (
            "Increase the annualised training rate. Review if learning opportunities are being offered "
            "equitably and whether the employee has had access to relevant upskilling pathways."
        ),
        "category": "career_growth",
        "evidence": (
            "Low training-per-year ratios signal neglect of career development, "
            "a key engagement driver (Saks, 2006)."
        ),
    },
    "Tenure_Performance_Ratio": {
        "action": (
            "Recognise long-tenure, high-performing employees with formal career milestone rewards "
            "and an accelerated progression plan. Stagnation for high performers is a primary "
            "flight risk signal."
        ),
        "category": "career_growth",
        "evidence": (
            "High tenure with flat progression and strong performance indicates an employee who may "
            "feel career-stuck, dramatically increasing departure probability (Allen et al., 2010)."
        ),
    },
    "Years_At_Company": {
        "action": (
            "For longer-tenured employees at plateau risk, consider lateral enrichment moves, "
            "stretch assignments, or a formal 'expert track' role to re-engage without requiring "
            "upward movement."
        ),
        "category": "career_growth",
        "evidence": (
            "Career plateau effects peak at 5–10 years of tenure without intervention "
            "(Feldman & Weitz, 1988)."
        ),
    },

    # ── ENGAGEMENT DOMAIN ─────────────────────────────────────────────────
    "Employee_Satisfaction_Score": {
        "action": (
            "Schedule an immediate structured 1:1 stay interview with the employee's manager. "
            "Identify specific dissatisfaction drivers. Build a personalised engagement action plan "
            "with a 30-day follow-up checkpoint."
        ),
        "category": "engagement",
        "evidence": (
            "Employee satisfaction is the strongest single-item predictor of retention intent "
            "(Harter et al., 2002; Gallup, 2022)."
        ),
    },
    "Performance_Score": {
        "action": (
            "For low performance, investigate root causes (workload? skill gap? manager conflict?) "
            "before applying a PIP. For high performers with low satisfaction, focus on recognition "
            "and career conversations rather than performance management."
        ),
        "category": "engagement",
        "evidence": (
            "Performance issues that are environmental (not individual) respond to structural "
            "changes rather than performance management escalation (Aguinis, 2019)."
        ),
    },
    "Remote_Work_Frequency": {
        "action": (
            "Offer the employee a formalised hybrid work arrangement that matches their stated "
            "preference. Avoid mandating full on-site presence for roles that can be performed remotely."
        ),
        "category": "engagement",
        "evidence": (
            "44% of employees report they would look for a new job if required to return to "
            "full-time in-office work (Microsoft Work Trend Index, 2022)."
        ),
    },
    "Sick_Days": {
        "action": (
            "Elevated sick-day usage is a leading indicator of burnout or disengagement. "
            "Initiate a confidential wellbeing conversation. Explore whether workload, workplace "
            "conflict, or personal circumstances are contributing."
        ),
        "category": "engagement",
        "evidence": (
            "Absenteeism patterns are a validated proxy for engagement levels and predict "
            "voluntary turnover with 6-month lead time (Farrell & Stamm, 1988)."
        ),
    },
    "Age": {
        "action": (
            "Tailor retention strategies to career life stage. Early-career employees prioritise "
            "growth and flexibility; mid-career employees prioritise compensation and progression; "
            "senior employees prioritise autonomy and legacy/purpose."
        ),
        "category": "engagement",
        "evidence": (
            "Generational and life-stage differences in work motivation require differentiated "
            "retention approaches (Twenge, 2010; Lyons & Kuron, 2014)."
        ),
    },

    # ── FALLBACK ──────────────────────────────────────────────────────────────
    "__fallback__": {
        "action": (
            "Schedule a structured stay interview with the employee's manager to understand "
            "specific concerns and co-create a personalised retention plan."
        ),
        "category": "engagement",
        "evidence": (
            "Stay interviews are among the most cost-effective retention tools, "
            "with a median implementation cost near zero (Finnegan, 2012)."
        ),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _assign_priority(shap_magnitude: float) -> str:
    """Assign intervention priority based on absolute SHAP value magnitude."""
    if shap_magnitude >= PRIORITY_CRITICAL:
        return "critical"
    elif shap_magnitude >= PRIORITY_RECOMMENDED:
        return "recommended"
    return "optional"


def generate_recommendations(
    top_risk_factors: List[Dict[str, Any]],
    max_recommendations: int = 5,
) -> List[Dict[str, Any]]:
    """
    Map SHAP top risk-increasing factors to structured HR interventions.

    Args:
        top_risk_factors: List of dicts with keys 'feature' and 'impact' (SHAP value).
                          Already produced by explanation_analysis.extract_local_explanation().
        max_recommendations: Maximum number of recommendations to return.

    Returns:
        List of recommendation dicts, sorted by priority (critical first).
    """
    recommendations = []
    seen_categories = set()

    for factor in top_risk_factors:
        feature = factor.get("feature", "")
        shap_val = abs(float(factor.get("impact", 0.0)))

        # Look up the feature — try exact match, then strip one-hot suffix
        entry = INTERVENTION_TABLE.get(feature)
        if entry is None:
            # Try stripping one-hot encoded suffix (e.g. "Department_Engineering" → no match → fallback)
            base = feature.rsplit("_", 1)[0] if "_" in feature else feature
            entry = INTERVENTION_TABLE.get(base)
        if entry is None:
            entry = INTERVENTION_TABLE["__fallback__"]

        priority = _assign_priority(shap_val)

        rec = {
            "feature": feature,
            "action": entry["action"],
            "category": entry["category"],
            "priority": priority,
            "shap_magnitude": round(shap_val, 4),
            "evidence": entry["evidence"],
        }
        recommendations.append(rec)

    # Sort: critical → recommended → optional, then by SHAP magnitude descending
    priority_order = {"critical": 0, "recommended": 1, "optional": 2}
    recommendations.sort(
        key=lambda r: (priority_order[r["priority"]], -r["shap_magnitude"])
    )

    return recommendations[:max_recommendations]


def summarise_for_prompt(
    employee_name: str,
    employee_metrics: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
) -> str:
    """
    Build a structured plain-text context block for injection into the Gemini prompt.

    Args:
        employee_name: Employee display name.
        employee_metrics: Key HR metrics dict (role, dept, risk score, satisfaction, etc.).
        recommendations: Output of generate_recommendations().

    Returns:
        Formatted string ready for LLM prompt injection.
    """
    lines = [
        f"EMPLOYEE: {employee_name}",
        f"Role: {employee_metrics.get('role', 'Unknown')}",
        f"Department: {employee_metrics.get('dept', 'Unknown')}",
        f"Attrition Risk Score: {employee_metrics.get('riskScore', 'N/A')}%",
        f"Risk Classification: {employee_metrics.get('riskLevelText', 'N/A')}",
        f"Satisfaction Score: {employee_metrics.get('satisfaction', 'N/A')}/10",
        f"Tenure: {employee_metrics.get('tenure', 'N/A')} years",
        f"Overtime (hrs/wk): {employee_metrics.get('overtime', 'N/A')}",
        f"Performance: {employee_metrics.get('performance', 'N/A')}",
        "",
        "EVIDENCE-BASED INTERVENTION RECOMMENDATIONS (from rule-based engine):",
    ]
    for i, rec in enumerate(recommendations, 1):
        lines.append(
            f"{i}. [{rec['priority'].upper()}] [{rec['category'].upper()}] {rec['action']}"
        )
        lines.append(f"   Evidence: {rec['evidence']}")

    return "\n".join(lines)
