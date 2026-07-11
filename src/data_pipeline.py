import pandas as pd
import numpy as np
import os
import logging
from statsmodels.tsa.stattools import adfuller

# FIX: Removed the space between % and (levelname)s to prevent format string errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Ingests raw Brent Oil pricing files, standardizes custom date vectors,
    handles missing records, and sorts chronologically to ensure time-series integrity.
    """
    if not os.path.exists(filepath):
        logging.error(f"Target data file not found at path: {filepath}")
        raise FileNotFoundError(f"Could not locate {filepath}")
        
    logging.info(f"Ingesting raw data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # 1. Clean column spacing and rename for consistency
    df.columns = [col.strip() for col in df.columns]
    
    # 2. FIX: Use format='mixed' to cleanly parse both '20-May-87' and 'Apr 22, 2020' formats
    logging.info("Parsing dates and verifying timeline structural order...")
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 3. Drop null values or gaps in trading sessions
    initial_len = len(df)
    df = df.dropna(subset=['Price'])
    final_len = len(df)
    
    if initial_len != final_len:
        logging.warning(f"Identified and removed {initial_len - final_len} rows with missing price entries.")
        
    logging.info(f"Data ingestion completed. Total clean trading records: {len(df)}")
    return df

def apply_log_transformations(df: pd.DataFrame, price_col: str = 'Price') -> pd.DataFrame:
    """
    Transforms raw absolute closing prices into continuous log returns 
    and moving volatility envelopes to isolate market regimes.
    """
    logging.info("Calculating continuous log return vectors and moving indicators...")
    
    # Calculate daily continuous log returns: ln(P_t / P_{t-1})
    df['Log_Return'] = np.log(df[price_col]) - np.log(df[price_col].shift(1))
    
    # Calculate 21-day standard rolling metrics (approx. 1 trading month)
    df['Rolling_Mean_21'] = df[price_col].rolling(window=21).mean()
    df['Rolling_Std_21'] = df[price_col].rolling(window=21).std()
    
    # Identify return statistical outliers outside the 3-sigma band
    df['Is_Outlier'] = np.where(np.abs(df['Log_Return']) > (3 * df['Log_Return'].std()), 1, 0)
    
    return df

def execute_adf_diagnostic(df: pd.DataFrame, target_col: str = 'Log_Return') -> dict:
    """
    Executes the Augmented Dickey-Fuller test to mathematically determine series stationarity.
    """
    clean_series = df[target_col].dropna()
    logging.info(f"Running Augmented Dickey-Fuller (ADF) test on column: {target_col}")
    
    result = adfuller(clean_series)
    
    diagnostics = {
        "Test_Statistic": result[0],
        "p_value": result[1],
        "Used_Lags": result[2],
        "Observations": result[3],
        "Critical_Values": result[4],
        "Stationary": result[1] < 0.05
    }
    
    print("\n" + "="*50)
    print(f" AUGMENTED DICKEY-FULLER DIAGNOSTICS FOR '{target_col}'")
    print("="*50)
    print(f"ADF Test Statistic : {diagnostics['Test_Statistic']:.6f}")
    print(f"Empirical p-value  : {diagnostics['p_value']:.4e}")
    print(f"Stationary Status  : {'PASS (Is Stationary)' if diagnostics['Stationary'] else 'FAIL (Non-Stationary)'}")
    print(f"Critical 5% Value  : {diagnostics['Critical_Values']['5%']:.4f}")
    print("="*50 + "\n")
    
    return diagnostics

if __name__ == "__main__":
    target_data_path = os.path.join("data", "raw", "BrentOilPrices.csv")
    
    try:
        processed_df = load_and_clean_data(target_data_path)
        processed_df = apply_log_transformations(processed_df)
        
        print("Evaluating raw pricing metrics:")
        execute_adf_diagnostic(processed_df, target_col='Price')
        
        print("Evaluating log returns metrics:")
        execute_adf_diagnostic(processed_df, target_col='Log_Return')
        
    except Exception as e:
        print(f"Execution failed with error configuration: {str(e)}")