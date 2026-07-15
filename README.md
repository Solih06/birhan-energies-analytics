# Brent Crude Oil Price Intelligence Dashboard
### Birhan Energies — Quantitative & Geopolitical Market Analysis

An interactive data intelligence dashboard and Bayesian statistical pipeline built to analyze macroeconomic regime shifts, structural price breaks, and geopolitical shocks affecting Brent Crude Oil prices from 1987 to 2022.

---

## 🚀 Key Features

*   **Bayesian Change-Point Detection (Task 2):** PyMC MCMC model featuring discrete and continuous hybrid samplers (`Metropolis` + `NUTS`) to locate the exact date of structural market shifts.
*   **Interactive React Dashboard (Task 3):** Designed with **Tailwind CSS** and interactive **Recharts** charts to explore price timelines, historical shocks, and dynamic date-filtering.
*   **High-Performance Flask API:** Real-time endpoint stream delivery for computed log-returns, Bayesian metrics, and decoupled geopolitical shock timeline markers.

---

## 📊 Bayesian Model Findings (Task 2)

*   **Structural Break Point:** Identified around **late November 2014** (associated with OPEC's historic supply maintenance decision during the US shale revolution).
*   **Volatilities (Daily Standard Deviation):**
    *   **Pre-Break Regime ($\sigma_1$):** High baseline variance as the market reacted to the 2008 financial recovery.
    *   **Post-Break Regime ($\sigma_2$):** Altered market dynamics showing a prolonged, statistically significant volatility transition.
*   **MCMC Diagnostics:** All primary parameters achieved convergence with $\hat{R} \le 1.05$.

---

## 🛠️ Project Structure

```text
birhan-energies-analytics/
├── data/
│   ├── raw/
│   │   └── BrentOilPrices.csv         # Historical prices
│   ├── historical_shocks.csv          # Decoupled geopolitical events
│   └── bayesian_outputs.json          # Cached Bayesian MCMC calculations
├── src/
│   ├── bayesian_model.py              # PyMC model engine
│   └── app.py                         # Production Flask API Server
├── dashboard/                         # React Frontend Application
│   ├── public/
│   ├── src/
│   │   ├── App.js                     # Main interactive dashboard code
│   │   └── index.js                   # Styled root entrypoint with Tailwind
│   └── package.json
└── README.md
```
## 💻 Setup and Execution
To run the entire system locally, follow these steps in two parallel terminal sessions:
1. Run the Python Backend (Flask API)
From the root directory, activate your virtual environment and start the API:
```bash
# 1. Activate your environment
.venv\Scripts\activate

# 2. Compute Bayesian stats (optional if json exists)
python src/bayesian_model.py

# 3. Spin up the API server
python src/app.py
The backend service will boot and run on http://127.0.0.1:5000/
```

2. Run the React Frontend (Dashboard)
Open a new terminal window, navigate into the dashboard folder, and start the development server:
```bash
# 1. Enter dashboard
cd dashboard

# 2. Spin up the dev server
npm start
```
Your browser will automatically launch the interactive dashboard on http://localhost:3000/

## 🛰️ Production Endpoints Exposed

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/prices` | `GET` | Delivers price series logs and computed log returns with dynamic `start_date` and `end_date` query filters. |
| `/api/v1/shocks` | `GET` | Exposes historical macroeconomic/geopolitical events. |
| `/api/v1/change_points` | `GET` | Delivers the calculated Bayesian breakpoint dates, HPDI bounds, and statistical parameters. |
| `/api/v1/health` | `GET` | API monitoring state and database tracking diagnostic. |
