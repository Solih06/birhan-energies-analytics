import os
import json
import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_bayesian_change_point():
    logging.info("Initializing baseline data ingestion for Bayesian Engine...")
    
    # Target transformed clean data structures
    processed_data_path = os.path.join("data", "raw", "BrentOilPrices.csv") 
    if not os.path.exists(processed_data_path):
        logging.error(f"Source price data missing at {processed_data_path}")
        return

    # Ingest price arrays and compute continuous log returns
    df = pd.read_csv(processed_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
    df = df.dropna().tail(2000) # Use a subset window to avoid local processor timeouts
    
    y_values = df['Log_Returns'].values
    n_records = len(y_values)
    
    logging.info(f"Configuring PyMC Model Architecture for {n_records} trading observations...")
    
    with pm.Model() as model:
        # Prior Distributions 
        # Latent switch point (tau) uniformly bounded across the row dimensions
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_records - 1)
        
        # Pre-break and post-break continuous distribution parameters
        mu_1 = pm.Normal("mu_1", mu=np.mean(y_values), sigma=0.1)
        mu_2 = pm.Normal("mu_2", mu=np.mean(y_values), sigma=0.1)
        
        sigma_1 = pm.Exponential("sigma_1", lam=1.0 / np.std(y_values))
        sigma_2 = pm.Exponential("sigma_2", lam=1.0 / np.std(y_values))
        
        # Mathematical Selector Switch Function
        idx = np.arange(n_records)
        mu = pm.math.switch(tau >= idx, mu_1, mu_2)
        sigma = pm.math.switch(tau >= idx, sigma_1, sigma_2)
        
        # Integrated Likelihood Array
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_values)
        
        # Execution of MCMC Sampling (NUTS + Metropolis step assignment)
        logging.info("Executing MCMC Sampling Simulations over 4 parallel chains...")
        step1 = pm.Metropolis(vars=[tau])
        step2 = pm.NUTS(vars=[mu_1, mu_2, sigma_1, sigma_2])
        
        idata = pm.sample(
            draws=4000, 
            tune=2000, 
            chains=4, 
            step=[step1, step2], 
            random_seed=42, 
            return_inferencedata=True
        )
        
    logging.info("Sampling complete. Running Convergence Diagnostics...")
    
    # Compute summary parameters and Gelman-Rubin (R-hat) tracking statistics
    summary = az.summary(idata, round_to=4)
    r_hat_values = summary['r_hat'].values
    
    # Explicitly check for successful chain mixing and stabilization
    if np.all(r_hat_values <= 1.05):
        logging.info("Convergence verified successfully: All parameters show R-hat <= 1.05.")
    else:
        logging.warning("Convergence warnings: Certain chains exhibit high R-hat values.")
        
    # Extract structural posterior density intervals (95% HPDI)
    tau_samples = idata.posterior["tau"].values.flatten()
    tau_mean = int(np.mean(tau_samples))
    
    hdi = az.hdi(idata, hdi_prob=0.95)
    tau_hdi_low = int(hdi["tau"][0])
    tau_hdi_high = int(hdi["tau"][1])
    
    change_point_date = str(df.iloc[tau_mean]['Date'].date())
    hdi_low_date = str(df.iloc[tau_hdi_low]['Date'].date())
    hdi_high_date = str(df.iloc[tau_hdi_high]['Date'].date())
    
    # Consolidate interpreted numerical outputs
    results_payload = {
        "tau_index": tau_mean,
        "change_point_date": change_point_date,
        "hpdi_95_lower_date": hdi_low_date,
        "hpdi_95_upper_date": hdi_high_date,
        "parameters": {
            "mu_1_pre_break": float(summary.loc["mu_1", "mean"]),
            "mu_2_post_break": float(summary.loc["mu_2", "mean"]),
            "sigma_1_pre_break": float(summary.loc["sigma_1", "mean"]),
            "sigma_2_post_break": float(summary.loc["sigma_2", "mean"])
        },
        "diagnostics": {
            "tau_r_hat": float(summary.loc["tau", "r_hat"]),
            "mu_1_r_hat": float(summary.loc["mu_1", "r_hat"]),
            "sigma_1_r_hat": float(summary.loc["sigma_1", "r_hat"])
        }
    }
    
    # Export pre-calculated statistical arrays to a committed asset path
    output_json_path = os.path.join("data", "bayesian_outputs.json")
    with open(output_json_path, "w") as f:
        json.dump(results_payload, f, indent=4)
        
    logging.info(f"Model outputs successfully saved to {output_json_path}")

if __name__ == "__main__":
    run_bayesian_change_point()