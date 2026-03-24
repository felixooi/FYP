import os
import shutil
import json
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env (GEMINI_API_KEY etc.)
load_dotenv()

# Import from the existing pipeline
from end_to_end_pipeline import run_inference_pipeline

# Import the rule-based recommendation engine (Phase D Layer 1)
from modules.recommendation_engine import generate_recommendations, summarise_for_prompt

app = FastAPI(title="RetentionAI Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
DATA_DIR = "data"
OUTPUT_DIR = "pipeline_output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Important: We serve the dashboard folder locally
app.mount("/dashboard", StaticFiles(directory="dashboard"), name="dashboard")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the single-file React dashboard at the root URL."""
    try:
        with open(os.path.join("dashboard", "index.html"), "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return HTMLResponse(content=f"Error loading dashboard: {str(e)}", status_code=500)

@app.post("/api/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    """
    1. Receives a CSV file from the frontend.
    2. Saves it as data/uploaded_dataset.csv.
    3. Triggers the end_to_end_pipeline script.
    4. Parses the output JSON and CSV and returns a single combined JSON object.
    """
    tmp_path = os.path.join(DATA_DIR, "uploaded_dataset.csv")
    
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run the existing data pipeline
        summary = run_inference_pipeline(
            input_file=tmp_path,
            output_dir=OUTPUT_DIR,
            model_path="models/best_model_tuned.pkl",
            metadata_path="models/tuning_metadata.json",
            selected_features_path="data/selected_features.json",
            scaler_path="data/scaler.pkl",
            fe_params_path="models/feature_engineering_params.json",
            run_xai=True,
            xai_local_count=None  # None = process ALL employees dynamically
        )
        
        # Load the predictions CSV so we can return the entire dataset
        # to the frontend (including predicted risk scores).
        predictions_path = os.path.join(OUTPUT_DIR, "inference_predictions.csv")
        df_preds = pd.read_csv(predictions_path)
        
        # The frontend prototype maps these keys for employees:
        # id, name, role, dept, location, riskScore, satisfaction, overtime, tenure, lastPromo, salaryRatio, salary, manager, performance
        # We need to map our dataset columns to these keys loosely.
        # This mapping assumes standard HR fields exist in the uploaded dataset.
        
        employees = []
        # Pre-compute per-department average salary for salary ratio calculation
        dept_avg_salary = df_preds.groupby("Department")["Monthly_Salary"].transform("mean") \
            if "Department" in df_preds.columns and "Monthly_Salary" in df_preds.columns \
            else None

        for index, row in df_preds.iterrows():
            # Extract probability or default
            prob = row.get("Attrition_Probability", 0)

            # Compute salary ratio vs department average (real data, not hardcoded)
            if dept_avg_salary is not None and dept_avg_salary.iloc[index] > 0:
                salary_ratio = round(float(row.get("Monthly_Salary", 0)) / float(dept_avg_salary.iloc[index]), 2)
            else:
                salary_ratio = 1.0  # Neutral fallback only if data missing

            # Map common columns or use fallbacks according to live_feed_inference_data.csv
            emp = {
                "id": index + 1,
                "name": row.get("Employee_Name", f"Employee {index + 1}"),
                "role": row.get("Job_Title", "Unknown Role"),
                "dept": row.get("Department", "Unknown Dept"),
                "location": row.get("Remote_Work_Frequency", "Unknown"),
                "riskScore": int(prob * 100),
                "satisfaction": int(row.get("Employee_Satisfaction_Score", 5)),
                "overtime": float(row.get("Overtime_Hours", 5)),
                "tenure": int(row.get("Years_At_Company", 0)),
                "lastPromo": int(row.get("Promotions", 0)),
                "salaryRatio": salary_ratio,
                "salary": float(row.get("Monthly_Salary", 50000)),
                "manager": row.get("Manager_Name", row.get("Manager", "N/A")),
                "performance": str(row.get("Performance_Score", 3))
            }
            # Attempt an exact Risk Level map if present
            if "Risk_Level" in row:
                emp["riskLevelText"] = row["Risk_Level"]

            employees.append(emp)

        # Load Time-Series Forecasting Model (Holt-Winters / ARIMA)
        import datetime
        from dateutil.relativedelta import relativedelta
        import random
        import pickle
        
        high_risk_count = sum(1 for e in employees if e["riskScore"] > 70)
        current_attrition_rate = int((high_risk_count / len(employees)) * 100) if employees else 0
        
        attrition_trend = []
        today = datetime.date.today()
        
        # Calculate an average risk multiplier to determine if the trend should go aggressively up or stay flat
        avg_risk = sum(e["riskScore"] for e in employees) / len(employees) if employees else 0
        growth_factor = 1.0 + ((avg_risk - 50) / 100.0)
        
        # Try to load the trained time-series model for robust trajectory shaping
        ts_model_path = os.path.join("models", "attrition_timeseries.pkl")
        ts_forecast = None
        try:
            if os.path.exists(ts_model_path):
                with open(ts_model_path, 'rb') as f:
                    ts_model = pickle.load(f)
                ts_forecast = ts_model.forecast(6).tolist()
        except Exception as e:
            print(f"Time-series model failed to load or forecast, falling back to heuristic math: {e}")

        # Generate NEXT 6 months trend starting at our current calculated rate
        running_rate = current_attrition_rate
        for i in range(6):
            month_date = today + relativedelta(months=i)
            month_name = month_date.strftime("%b")
            
            if i == 0:
                val = running_rate
            else:
                if ts_forecast is not None and len(ts_forecast) >= 6:
                    # Apply the EXACT seasonal delta shape predicted by the Time-Series model
                    # So if the model predicts rates go from 15.0 -> 16.5, that's a +10% relative increase
                    # We apply that exact +10% relative increase to our `running_rate`
                    relative_delta = (ts_forecast[i] - ts_forecast[i-1]) / ts_forecast[i-1]
                    running_rate = running_rate * (1.0 + relative_delta)
                    # We also blend in the dataset's specific growth factor (so high-risk datasets trend higher)
                    running_rate = running_rate * (1.0 + max(0, growth_factor - 1.0)*0.2)
                else:
                    # Fallback math if model is missing
                    variance = random.uniform(-2.0, 3.0) 
                    running_rate = running_rate * growth_factor + variance
                    
                val = max(5, int(running_rate)) # Floor at 5%
                
            attrition_trend.append({"month": month_name, "value": val})

        # Build response
        response = {
            "status": "success",
            "pipeline_summary": summary,
            "employees": employees,
            "attrition_trend": attrition_trend
        }
        
        # Attach any XAI local explanations to the respective employees
        if "xai" in summary and "local_explanation_files" in summary["xai"]:
            for ex_file in summary["xai"]["local_explanation_files"]:
                try:
                    with open(ex_file, "r") as f:
                        exp_data = json.load(f)
                        emp_idx = exp_data.get("employee_index")
                        # Attach the explanation text to the employee object
                        if emp_idx is not None and emp_idx < len(employees):
                            employees[emp_idx]["xai_explanation"] = exp_data.get("explanation_text")
                            employees[emp_idx]["xai_factors_pos"] = exp_data.get("top_risk_increasing_factors")
                            employees[emp_idx]["xai_factors_neg"] = exp_data.get("top_risk_reducing_factors")
                except Exception as e:
                    print(f"Error loading explanation file {ex_file}: {e}")

        return JSONResponse(content=response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


from pydantic import BaseModel

class SimulationParams(BaseModel):
    salary_increase_pct: float = 0.0       # e.g., 10.0 = +10% of Monthly_Salary
    overtime_reduction_hrs: float = 0.0    # e.g., 5.0 = reduce Overtime_Hours by 5
    mentorship_enabled: bool = False        # e.g., True = +1 satisfaction for eligible employees
    training_hours_increase: float = 0.0   # e.g., 8.0 = add 8hrs of training per quarter


@app.post("/api/simulate")
async def simulate_intervention(params: SimulationParams):
    """
    What-If Simulation Engine.
    Applies proposed HR interventions to the uploaded dataset and re-runs 
    predictions through the trained ML model to calculate true risk deltas.
    """
    uploaded_path = os.path.join(DATA_DIR, "uploaded_dataset.csv")
    if not os.path.exists(uploaded_path):
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": "No dataset uploaded yet. Please upload and analyze a CSV first via /api/analyze."
        })

    try:
        # Load artifacts (re-uses same paths as analyze)
        from end_to_end_pipeline import (
            _load_artifacts, _preprocess_for_inference, _predict_with_model
        )
        model, threshold, selected_features, scaler, fe_params = _load_artifacts(
            model_path="models/best_model_tuned.pkl",
            metadata_path="models/tuning_metadata.json",
            selected_features_path="data/selected_features.json",
            scaler_path="data/scaler.pkl",
            fe_params_path="models/feature_engineering_params.json",
        )

        # Load the original uploaded data
        df_original = pd.read_csv(uploaded_path)

        # --- Run original prediction for baseline ---
        X_orig, _ = _preprocess_for_inference(df_original, selected_features, scaler, fe_params)
        y_prob_orig = _predict_with_model(model, X_orig)

        # --- Apply simulated interventions to a copy ---
        df_sim = df_original.copy()
        if params.salary_increase_pct > 0 and "Monthly_Salary" in df_sim.columns:
            df_sim["Monthly_Salary"] = df_sim["Monthly_Salary"] * (1 + params.salary_increase_pct / 100.0)
        if params.overtime_reduction_hrs > 0 and "Overtime_Hours" in df_sim.columns:
            df_sim["Overtime_Hours"] = (df_sim["Overtime_Hours"] - params.overtime_reduction_hrs).clip(lower=0)
        if params.mentorship_enabled and "Employee_Satisfaction_Score" in df_sim.columns:
            # Mentorship modelled as a satisfaction increase capped at max scale (10)
            df_sim["Employee_Satisfaction_Score"] = (df_sim["Employee_Satisfaction_Score"] + 0.8).clip(upper=10)
        if params.training_hours_increase > 0 and "Training_Hours" in df_sim.columns:
            df_sim["Training_Hours"] = df_sim["Training_Hours"] + params.training_hours_increase

        # --- Run simulated prediction ---
        X_sim, _ = _preprocess_for_inference(df_sim, selected_features, scaler, fe_params)
        y_prob_sim = _predict_with_model(model, X_sim)

        # --- Build diff output ---
        results = []
        for i in range(len(df_original)):
            orig_risk = int(y_prob_orig[i] * 100)
            sim_risk = int(y_prob_sim[i] * 100)
            results.append({
                "employee_index": i,
                "employee_name": str(df_original.iloc[i].get("Employee_Name", f"Employee {i+1}")),
                "department": str(df_original.iloc[i].get("Department", "Unknown")),
                "original_risk": orig_risk,
                "simulated_risk": sim_risk,
                "delta": sim_risk - orig_risk,  # Negative = improvement
            })

        # Aggregate stats
        avg_original = round(sum(r["original_risk"] for r in results) / len(results), 1)
        avg_simulated = round(sum(r["simulated_risk"] for r in results) / len(results), 1)
        high_risk_original = sum(1 for r in results if r["original_risk"] > 70)
        high_risk_simulated = sum(1 for r in results if r["simulated_risk"] > 70)

        return JSONResponse(content={
            "status": "success",
            "simulation_params": params.model_dump(),
            "aggregate": {
                "avg_risk_before": avg_original,
                "avg_risk_after": avg_simulated,
                "avg_delta": round(avg_simulated - avg_original, 1),
                "high_risk_count_before": high_risk_original,
                "high_risk_count_after": high_risk_simulated,
                "high_risk_reduction": high_risk_original - high_risk_simulated,
            },
            "employee_results": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

class RecommendRequest(BaseModel):
    employee_id: int                  # 0-based index from the employees list
    employee_name: str = ""
    role: str = ""
    dept: str = ""
    risk_score: int = 0
    risk_level_text: str = ""
    satisfaction: float = 5.0
    tenure: int = 0
    overtime: float = 0.0
    performance: str = "3"


@app.post("/api/recommend")
async def generate_retention_recommendation(req: RecommendRequest):
    """
    Phase D — Attrition Mitigation Recommendation Engine.

    Layer 1: Maps SHAP top risk-increasing factors to evidence-based HR interventions.
    Layer 2: Calls Gemini 2.5 Flash to generate a personalised retention narrative.
    """
    # ── Locate the XAI explanation file written by the pipeline ───────────────
    local_exp_path = os.path.join(
        OUTPUT_DIR, "xai", "local", f"employee_{req.employee_id}_explanation.json"
    )

    top_risk_factors = []
    precomputed_recommendations = None
    if os.path.exists(local_exp_path):
        try:
            with open(local_exp_path, "r") as f:
                exp_data = json.load(f)
            top_risk_factors = exp_data.get("top_risk_increasing_factors", [])
            # Prefer pre-computed recommendations embedded by the pipeline (Phase D)
            precomputed_recommendations = exp_data.get("rule_recommendations")
        except Exception as e:
            print(f"Warning: could not load explanation file: {e}")

    # ── Layer 1: Rule-based recommendations ───────────────────────────────────
    # Use pre-computed output from the pipeline run if available;
    # otherwise recompute on-the-fly (e.g., older pipeline run or missing file).
    if precomputed_recommendations is not None:
        rule_recommendations = precomputed_recommendations
    else:
        # Remap key 'shap_value' → 'impact' for the engine (matches pipeline format)
        factors_for_engine = [
            {"feature": f.get("feature", ""), "impact": f.get("shap_value", 0.0)}
            for f in top_risk_factors
        ]
        rule_recommendations = generate_recommendations(
            top_risk_factors=factors_for_engine,
            max_recommendations=5,
        )

    # ── Layer 2: Gemini GenAI narrative ───────────────────────────────────────
    llm_narrative = None
    gemini_error = None
    try:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        employee_metrics = {
            "role": req.role,
            "dept": req.dept,
            "riskScore": req.risk_score,
            "riskLevelText": req.risk_level_text,
            "satisfaction": req.satisfaction,
            "tenure": req.tenure,
            "overtime": req.overtime,
            "performance": req.performance,
        }

        context_block = summarise_for_prompt(
            employee_name=req.employee_name,
            employee_metrics=employee_metrics,
            recommendations=rule_recommendations,
        )

        prompt = f"""You are an expert HR Business Partner with 15 years of experience in talent retention.

Below is a structured risk profile and evidence-based intervention recommendations for an at-risk employee generated by an AI attrition prediction system.

{context_block}

Using the above structured recommendations as the foundation, generate the following three outputs:

1. EXECUTIVE SUMMARY (2-3 sentences): A concise, non-technical risk summary for senior HR leadership, explaining why this employee's attrition risk is elevated and what the highest-priority intervention domain is.

2. MANAGER CONVERSATION STARTERS (3-4 bullet points): Specific, psychologically-informed talking points for the line manager's next 1:1 with this employee. These should be open-ended questions that surface the root cause without being accusatory.

3. DRAFT RETENTION EMAIL (subject line + body): A professional, empathetic email from the HR Business Partner to the employee inviting them to a confidential career conversation. Tone should be supportive, not alarming.

Format your response with clear section headers: ## Executive Summary, ## Manager Conversation Starters, ## Draft Email."""

        response = model.generate_content(prompt)
        llm_narrative = response.text

    except Exception as e:
        gemini_error = str(e)
        print(f"Gemini API error: {e}")

    return JSONResponse(content={
        "status": "success",
        "employee_id": req.employee_id,
        "employee_name": req.employee_name,
        "rule_recommendations": rule_recommendations,
        "llm_narrative": llm_narrative,
        "gemini_error": gemini_error,
    })


if __name__ == "__main__":
    import uvicorn
    # Make sure this runs from the root of the project
    uvicorn.run("app:app", host="0.0.0.0", port=8050, reload=True)
