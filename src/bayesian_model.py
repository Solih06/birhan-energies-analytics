import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from src.data_pipeline import load_and_clean_data
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="  # Instructs PyTensor to avoid searching for missing system g++ compilers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_and_sample_change_point_model(df: pd.DataFrame, value_col: str = 'Price', samples: int = 1500):
    """
    Constructs a Bayesian Change Point model using PyMC to isolate an unobserved shift parameter (tau).
    Uses a switch condition to evaluate data distribution regimes before and after the transition.
    """
    logging.info(f"Preparing Bayesian inference model for column: {value_col}")
    
    # Extract values and create a sequential time index vector
    data_values = df[value_col].values
    n_data_points = len(data_values)
    time_idx = np.arange(n_data_points)
    
    # Define empirical hyper-priors based on data properties to assist MCMC convergence
    mean_prior = data_values.mean()
    std_prior = data_values.std()
    
    with pm.Model() as model:
        # 1. Define Tau: The discrete switch point uniform prior across the time timeline
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_data_points - 1)
        
        # 2. Define early and late regime expectations (Mu_1 and Mu_2)
        mu_1 = pm.Normal("mu_1", mu=mean_prior, sigma=std_prior)
        mu_2 = pm.Normal("mu_2", mu=mean_prior, sigma=std_prior)
        
        # 3. Define observation noise prior (Sigma)
        sigma = pm.Exponential("sigma", lam=1.0 / std_prior)
        
        # 4. Construct the mathematical switch condition based on index vs tau position
        # If time_idx < tau, select mu_1, else select mu_2
        mu_assigned = pm.math.switch(tau > time_idx, mu_1, mu_2)
        
        # 5. Link priors to true historical observations via the likelihood profile
        likelihood = pm.Normal("likelihood", mu=mu_assigned, sigma=sigma, observed=data_values)
        
        # 6. Execute Markov Chain Monte Carlo sampler
        logging.info(f"Initiating MCMC sampling sequence ({samples} draws)...")
        trace = pm.sample(draws=samples, tune=1000, return_inferencedata=True, random_seed=42)
        
    logging.info("MCMC sampling sequence completed successfully.")
    return trace

def plot_and_save_bayesian_results(trace, df: pd.DataFrame, output_image: str = "notebooks/bayesian_switch_results.png"):
    """
    Extracts posterior summaries and visualizes the calculated change point location probability distribution.
    """
    logging.info("Extracting posterior samples and compiling diagnostic visualizations...")
    
    # Extract calculated switch points out of the inference tracking trace
    posterior_tau = trace.posterior["tau"].values.flatten()
    
    # Calculate the most probable index point (the mode)
    calculated_mode_idx = int(pd.Series(posterior_tau).mode()[0])
    detected_date = df['Date'].iloc[calculated_mode_idx].strftime('%Y-%m-%d')
    
    plt.figure(figsize=(12, 6))
    plt.hist(posterior_tau, bins=50, density=True, color='purple', alpha=0.6, label='Posterior Distribution of $\\tau$')
    plt.axvline(calculated_mode_idx, color='red', linestyle='--', linewidth=2, label=f'Identified Change Point: {detected_date}')
    
    plt.title("Bayesian Change Point Detection — Posterior Probability Density of $\\tau$")
    plt.xlabel("Timeline Trading Index Location")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure parent folder paths exist before writing image binary arrays
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image)
    plt.close()
    
    print(f"\n" + "="*50)
    print(f" 🚀 BAYESIAN CHANGE POINT IDENTIFIED")
    print("="*50)
    print(f"Index Location   : {calculated_mode_idx}")
    print(f"Estimated Date   : {detected_date}")
    print(f"Result Visualization Saved To: {output_image}")
    print("="*50 + "\n")
    
    return detected_date

if __name__ == "__main__":
    target_data_path = os.path.join("data", "raw", "BrentOilPrices.csv")
    
    if os.path.exists(target_data_path):
        # Load and take a slice of the recent dataset for sampling performance
        full_df = load_and_clean_data(target_data_path)
        
        # Slice the last 1000 records (e.g., recent decade market shifts) to prevent CPU resource limits
        analysis_slice = full_df.tail(1000).reset_index(drop=True)
        
        try:
            mcmc_trace = build_and_sample_change_point_model(analysis_slice, value_col='Price')
            plot_and_save_bayesian_results(mcmc_trace, analysis_slice)
        except Exception as e:
            logging.error(f"Bayesian modeling routine hit an error: {str(e)}")
    else:
        print(f"Could not initialize execution. File not found at: {target_data_path}")