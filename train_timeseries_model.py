import os
import pandas as pd
import numpy as np
import pickle
import datetime
import warnings
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")

# Setup paths
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "attrition_timeseries.pkl")

def generate_synthetic_historical_data(months=24, base_rate=15.0):
    """
    Generate a 24-month synthetic historical monthly attrition rate dataset.
    Follows Additive Model: Output = Base + Trend + Seasonality + Noise
    """
    today = datetime.date.today()
    dates = []
    rates = []
    
    # Let's say the trend is slowly increasing over the last 2 years 
    trend_factor = np.linspace(-3.0, 1.5, months)
    
    # Seasonality: Higher in March/April (post bonuses), Lower in Nov/Dec
    month_modifiers = {
        1: 1.0, 2: 0.5, 3: 3.0, 4: 2.5, 
        5: 1.0, 6: 0.5, 7: 1.5, 8: 1.0, 
        9: 2.0, 10: 0.0, 11: -1.5, 12: -2.0
    }
    
    for i in range(months-1, -1, -1):
        dt = today - relativedelta(months=i)
        # Normalize to the 1st of the month for time-series frequency alignment
        dt = dt.replace(day=1)
        dates.append(dt)
        
        # Base + Trend + Seasonality + Noise
        seasonality = month_modifiers[dt.month]
        noise = np.random.normal(0, 1.2) # Std deviation of 1.2% variance
        
        rate = base_rate + trend_factor[months - 1 - i] + seasonality + noise
        rate = max(2.0, rate) # Floor the rate at 2% minimum
        rates.append(rate)
        
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Attrition_Rate': rates
    })
    df.set_index('Date', inplace=True)
    df = df.asfreq('MS') # Set explicitly to Monthly Start frequency
    return df

def train_and_save_model():
    print("Generating 24-month synthetic historical attrition dataset...")
    df = generate_synthetic_historical_data(months=24, base_rate=14.0)
    
    print("\nSample Historical Data (Last 5 Months):")
    print(df.tail())
    
    print("\nTraining Holt-Winters Exponential Smoothing Model...")
    # Additive trend and additive seasonality (period=12 months)
    model = ExponentialSmoothing(
        df['Attrition_Rate'], 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12,
        initialization_method='estimated'
    ).fit(optimized=True)
    
    print("\nModel trained successfully. Exporting to:", MODEL_PATH)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    # Test a 6-month forecast
    forecast = model.forecast(6)
    print("\n6-Month Forecast Preview (Unadjusted):")
    print(forecast)

if __name__ == "__main__":
    train_and_save_model()
