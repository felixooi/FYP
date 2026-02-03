"""
Simple Interactive Prediction Script
Allows you to manually set employee features and get attrition predictions.
"""

import pandas as pd
import joblib
import json

# Load the trained model
print("Loading model...")
model = joblib.load('models/best_model.pkl')
print("Model loaded successfully!\n")

# Load selected features (model expects this exact schema)
with open('data/selected_features.json', 'r') as f:
    SELECTED_FEATURES = json.load(f)

def build_model_input(feature_series: pd.Series) -> pd.DataFrame:
    """
    Align a feature Series to the model's expected columns.
    Missing features are filled with 0. Extra features are dropped.
    Returns a single-row DataFrame with correct column order.
    """
    row = feature_series.to_frame().T
    # Add missing columns
    for col in SELECTED_FEATURES:
        if col not in row.columns:
            row[col] = 0
    # Drop extras and enforce order
    row = row[SELECTED_FEATURES]
    return row

# Load a sample employee to see the feature structure
sample_data = pd.read_csv('data/test_data.csv')
sample_employee = sample_data.drop(columns=['Resigned']).iloc[0]

print("="*80)
print("EMPLOYEE ATTRITION PREDICTION TESTING")
print("="*80)

# Display feature names and sample values
if False:
    print("\nFeatures in the model (45 total):")
    print("-"*80)
    for i, (feature, value) in enumerate(sample_employee.items(), 1):
        print(f"{i:2d}. {feature:35s} = {value:.4f}")

    print("\n" + "="*80)
    print("OPTION 1: Predict with Sample Employee (as-is)")
    print("="*80)

# Predict with sample employee
    sample_input = build_model_input(sample_employee)
    prediction = model.predict(sample_input)[0]
    probability = model.predict_proba(sample_input)[0]

    print(f"\nPrediction: {'Will RESIGN' if prediction == 1 else 'Will STAY'}")
    print(f"Attrition Probability: {probability[1]:.2%}")
    print(f"Retention Probability: {probability[0]:.2%}")
    print(f"Risk Score: {probability[1]*100:.1f}/100")

    if probability[1] < 0.3:
        risk_level = "LOW RISK"
    elif probability[1] < 0.7:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "HIGH RISK"
    print(f"Risk Level: {risk_level}")

    print("\n" + "="*80)
    print("OPTION 2: Modify Features and Predict")
    print("="*80)

# Create a modifiable employee (copy of sample)
custom_employee = sample_employee.copy()

# Example modifications you can make:
#print("\nExample: Let's modify some key features...")
#print("-"*80)

# Modify key features (you can change these values)
custom_employee['Monthly_Salary'] = 0  
custom_employee['Employee_Satisfaction_Score'] = 0 
custom_employee['Overtime_Hours'] = 3.5  
custom_employee['Work_Hours_Per_Week'] = 3.5 
custom_employee['Performance_Score'] = 0.5 
print("Modified features:")
print(f"  Monthly_Salary: {custom_employee['Monthly_Salary']:.4f}")
print(f"  Employee_Satisfaction_Score: {custom_employee['Employee_Satisfaction_Score']:.4f}")
print(f"  Overtime_Hours: {custom_employee['Overtime_Hours']:.4f}")
print(f"  Work_Hours_Per_Week: {custom_employee['Work_Hours_Per_Week']:.4f}")
print(f"  Performance_Score: {custom_employee['Performance_Score']:.4f}")

# Predict with modified employee
custom_input = build_model_input(custom_employee)
prediction_custom = model.predict(custom_input)[0]
probability_custom = model.predict_proba(custom_input)[0]

print(f"\nPrediction: {'Will RESIGN' if prediction_custom == 1 else 'Will STAY'}")
print(f"Attrition Probability: {probability_custom[1]:.2%}")
print(f"Retention Probability: {probability_custom[0]:.2%}")
print(f"Risk Score: {probability_custom[1]*100:.1f}/100")

if probability_custom[1] < 0.3:
    risk_level_custom = "LOW RISK"
elif probability_custom[1] < 0.7:
    risk_level_custom = "MEDIUM RISK"
else:
    risk_level_custom = "HIGH RISK"
print(f"Risk Level: {risk_level_custom}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
#print(f"Original Employee: {probability[1]:.2%} attrition risk ({risk_level})")
print(f"Modified Employee: {probability_custom[1]:.2%} attrition risk ({risk_level_custom})")
#print(f"Change: {(probability_custom[1] - probability[1])*100:+.1f} percentage points")


# Customization instructions:
# To test your own scenarios, modify the values in the script:
# Note: Values are scaled (standardized), so:
#   - 0 = average
#   - Positive = above average
#   - Negative = below average
#   - Typical range: -3 to +3

print("\n" + "="*80)
print("OPTION 3: Create Employee from Scratch")
print("="*80)

# Create a completely custom employee (only the 20 selected features)
# Note: Values are scaled (standardized), so:
#   - 0 = average
#   - Positive = above average
#   - Negative = below average
custom_from_scratch = pd.Series({
    'Employee_Satisfaction_Score': 0.5,  # High satisfaction
    'Burnout_Risk': 4.0,  # Low burnout risk
    'Workload_Intensity': 3.5,  # Below average workload
    'Performance_Score': 1.0,  # Above average performance
    'Work_Hours_Per_Week': 3.0,  # Average work hours
    'Training_Hours': 1.5,  # Above average training
    'Work_Life_Balance': -2.5,  # Good work-life balance
    'Salary_Performance_Gap': 0.5,  # Fair compensation
    'Training_Per_Year': 1.0,  # Good training
    'Project_Load': 4.0,  # Average project load
    'Tenure_Performance_Ratio': 0.5,  # Good ratio
    'Overtime_Ratio': 3.0,  # Low overtime ratio
    'Promotions': -1.0,  # Average promotions
    'Projects_Handled': 2.0,  # Average projects
    'Is_Underutilized': 1.5,  # Not underutilized
    'Overtime_Hours': 3.0,  # Below average overtime
    'Is_Overworked': 2.0,  # Not overworked
    'Sick_Days': 2.0,  # Average sick days
    'Monthly_Salary': 0.5,  # Above average salary
    'Age': 1.5,  # Slightly above average age
})

# Predict with custom employee
scratch_input = build_model_input(custom_from_scratch)
prediction_scratch = model.predict(scratch_input)[0]
probability_scratch = model.predict_proba(scratch_input)[0]

print("P(resign) =", probability_scratch[1])
print(f"\nPrediction: {'Will RESIGN' if prediction_scratch == 1 else 'Will STAY'}")
print(f"Attrition Probability: {probability_scratch[1]:.2%}")
print(f"Risk Score: {probability_scratch[1]*100:.1f}/100")

if probability_scratch[1] < 0.3:
    print(f"Risk Level: LOW RISK")
elif probability_scratch[1] < 0.7:
    print(f"Risk Level: MEDIUM RISK")
else:
    print(f"Risk Level: HIGH RISK")
