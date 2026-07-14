import os
import json
import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_production_bayesian_model():
    logging.info("Initiating ingestion of historical asset arrays for Task 2...")
    
    raw_data_path = os.path.join("data", "raw", "BrentOilPrices.csv")
    if not os.path.exists(raw_data_path):
        logging.error(f"Execution terminated: Absolute price asset missing at {raw_data_path}")
        return

    # Ingest data and compute stationary daily log returns
    df = pd.read_csv(raw_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
    df = df.dropna()
    
    # We slice a dense multi-year target window (e.g., last 2500 trading sessions) 
    # to evaluate structural changes without hitting local memory constraints
    df_window = df.tail(2500).reset_index(drop=True)
    y_values = df_window['Log_Returns'].values
    n_records = len(y_values)
    
    logging.info(f"Configuring complete PyMC architecture over {n_records} rows...")
    
    with pm.Model() as model:
        # Task 2 Requirement: Define true joint probability priors
        # Bounding latent break index tau uniformly across the time matrix dimension
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_records - 1)
        
        # Pre-break and post-break means centered around empirical returns
        mu_1 = pm.Normal("mu_1", mu=np.mean(y_values), sigma=0.1)
        mu_2 = pm.Normal("mu_2", mu=np.mean(y_values), sigma=0.1)
        
        # Exponential positive-bounded priors for scaling parameter regimes
        sigma_1 = pm.Exponential("sigma_1", lam=1.0 / np.std(y_values))
        sigma_2 = pm.Exponential("sigma_2", lam=1.0 / np.std(y_values))
        
        # Task 2 Requirement: Deterministic index execution switch selector function
        idx = np.arange(n_records)
        mu = pm.math.switch(tau >= idx, mu_1, mu_2)
        sigma = pm.math.switch(tau >= idx, sigma_1, sigma_2)
        
        # Integrated Normal Likelihood binding stationary log returns
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_values)
        
        # Task 2 Requirement: Explicit step sampler mapping configuration
        step_discrete = pm.Metropolis(vars=[tau])
        step_continuous = pm.NUTS(vars=[mu_1, mu_2, sigma_1, sigma_2])
        
        logging.info("Running MCMC simulation across 4 parallel sampling chains...")
        idata = pm.sample(
            draws=4000, 
            tune=2000, 
            chains=4, 
            step=[step_discrete, step_continuous], 
            random_seed=42,
            return_inferencedata=True
        )
        
    logging.info("Sampling complete. Running rigorous convergence check diagnostics...")
    
    # Task 2 Requirement: Formulate exact Gelman-Rubin summary statistics
    summary = az.summary(idata, round_to=4)
    logging.info(f"\n{summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']]}")
    
    # Check if chains have correctly mixed and converged
    tau_rhat = summary.loc["tau", "r_hat"]
    if tau_rhat <= 1.05:
        logging.info(f"Convergence successfully verified. Tau R-hat: {tau_rhat}")
    else:
        logging.warning(f"Convergence anomaly detected: Tau R-hat stands at {tau_rhat}")
        
    # Translate structural mean indices back into string calendar dates
    tau_mean_idx = int(np.floor(summary.loc["tau", "mean"]))
    hdi = az.hdi(idata, hdi_prob=0.95)
    tau_hdi_low = int(np.floor(hdi["tau"][0]))
    tau_hdi_high = int(np.floor(hdi["tau"][1]))
    
    # Guard index ranges
    tau_mean_idx = min(max(0, tau_mean_idx), n_records - 1)
    tau_hdi_low = min(max(0, tau_hdi_low), n_records - 1)
    tau_hdi_high = min(max(0, tau_hdi_high), n_records - 1)
    
    calculated_break_date = str(df_window.iloc[tau_mean_idx]['Date'].date())
    hpdi_lower_date = str(df_window.iloc[tau_hdi_low]['Date'].date())
    hpdi_upper_date = str(df_window.iloc[tau_hdi_high]['Date'].date())
    
    # Assemble complete concrete numerical payload
    results_payload = {
        "status": "calculated",
        "sampled_records": n_records,
        "change_point_index": tau_mean_idx,
        "calculated_break_date": calculated_break_date,
        "hpdi_95_lower_date": hpdi_lower_date,
        "hpdi_95_upper_date": hpdi_upper_date,
        "parameters_before_break": {
            "mu_1_mean_return": float(summary.loc["mu_1", "mean"]),
            "sigma_1_volatility": float(summary.loc["sigma_1", "mean"])
        },
        "parameters_after_break": {
            "mu_2_mean_return": float(summary.loc["mu_2", "mean"]),
            "sigma_2_volatility": float(summary.loc["sigma_2", "mean"])
        },
        "convergence_diagnostics": {
            "tau_r_hat": float(summary.loc["tau", "r_hat"]),
            "mu_1_r_hat": float(summary.loc["mu_1", "r_hat"]),
            "sigma_1_r_hat": float(summary.loc["sigma_1", "r_hat"])
        }
    }
    
    output_path = os.path.join("data", "bayesian_outputs.json")
    with open(output_path, "w") as f:
        json.dump(results_payload, f, indent=4)
        
    logging.info(f"Concrete Bayesian numerical outputs successfully saved to {output_path}")

if __name__ == "__main__":
    run_production_bayesian_model()