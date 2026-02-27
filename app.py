import os
import shutil
import json
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import from the existing pipeline
from end_to_end_pipeline import run_inference_pipeline

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
            xai_local_count=20  # increased to pull explanations for the dashboard
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
        for index, row in df_preds.iterrows():
            # Extract probability or default
            prob = row.get("Attrition_Probability", 0)
            
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
                "salaryRatio": 1.0, # Salary Ratio doesn't have an exact match in the new CSV
                "salary": float(row.get("Monthly_Salary", 50000)),
                "manager": "Unknown Manager",
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

if __name__ == "__main__":
    import uvicorn
    # Make sure this runs from the root of the project
    uvicorn.run("app:app", host="0.0.0.0", port=8050, reload=True)
