# 📋 Statistical Assumptions, Limitations, and Causality Caveats

## 1. Core Workflow Assumptions

### A. Statistical Grounding
* **Log-Transform Stationarity:** The primary pricing time series exhibits strong trend dependency and non-stationarity in its raw absolute form ($P_t$). Downstream statistical inference assumes that converting values into continuous log returns ($r_t = \ln(P_t / P_{t-1})$) resolves unit roots, a condition mathematically validated via the Augmented Dickey-Fuller (ADF) tests.
* **Variance Regime Stability:** Rolling indicators (e.g., 21-day standard deviations) assume variance maps are locally bounded, though oil markets consistently exhibit long memory and GARCH-style volatility clustering behavior.

### B. Bayesian Change Point Modeling (PyMC)
* **Discrete Parameter Mapping ($\tau$):** The Bayesian model assumes changes happen at distinct, sudden structural moments rather than through long, continuous processes.
* **Uniform Prior Constraints:** The change point parameter $\tau$ is bounded using a discrete uniform prior spanning across the temporal array ($0, N-1$), assuming an equal priori probability for structural changes at any given point in the timeline.

---

## 2. Analytical Limitations

### A. Temporal Slicing & Sampling Granularity
* **Sampling Window Compression:** To prevent MCMC processor timeouts and compilation constraints on local environments, the model samples a targeted slice of the data (e.g., 1,000 trailing trading sessions). Structural changes occurring prior to this window are omitted.
* **Omission of Non-Trading Days:** Data arrays capture weekday market closing prices. Weekend structural developments are delayed until the subsequent market opening, causing localized return jumps.

### B. Algorithmic Bounds
* **Single Change Point Constraint:** The current model evaluates a single structural transition step ($\tau$). In reality, commodities like crude oil undergo multiple regime transformations driven by recurring geopolitical cycles.

---

## 3. Causality Caveats (Correlation vs. Causality)

* **Exogenous Identification Risks:** Matching an identified structural change point ($\tau$) with an external global shock (e.g., the 2020 pandemic declaration or 2022 conflict) is **associative, not strictly causal**. 
* **Omitted Variable Bias (OVB):** The pipeline tracks price vectors in isolation. It does not control for simultaneous macroeconomic variables—such as Federal Reserve interest rate hikes, US Dollar strength fluctuations (DXY index), or global maritime shipping container logistics capacity indices. 
* **Lead-Lag Asymmetry:** Markets often price in anticipated macroeconomic changes before they officially occur (e.g., trading futures ahead of an OPEC announcement), which can create a timing gap between an identified model change point and the physical event date.